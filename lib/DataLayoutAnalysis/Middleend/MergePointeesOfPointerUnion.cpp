//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_set>

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/RecursiveCoroutine.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLAStep.h"

using namespace llvm;

static Logger<> Log("dla-merge-pointees-of-ptr-union");

namespace dla {

using LTSN = LayoutTypeSystemNode;
using NeighborsConstIterator = LTSN::NeighborsSet::const_iterator;

static bool hasOutgoingPointerEdge(const LayoutTypeSystemNode *N) {
  using CPointerT = EdgeFilteredGraph<const dla::LayoutTypeSystemNode *,
                                      dla::isPointerEdge>;
  using PointerGraph = llvm::GraphTraits<CPointerT>;
  auto It = PointerGraph::child_begin(N);
  auto End = PointerGraph::child_end(N);
  return It != End;
};

static bool isWellFormedPointer(const LTSN *Pointer) {
  return Pointer->Successors.size() == 1 and hasOutgoingPointerEdge(Pointer);
}

static LTSN *getPointee(LTSN *Pointer) {
  revng_assert(isWellFormedPointer(Pointer));
  return Pointer->Successors.begin()->first;
}

bool MergePointeesOfPointerUnion::runOnTypeSystem(LayoutTypeSystem &TS) {
  bool Changed = false;

  revng_log(Log, "MergePointeesOfPointerUnion");
  LoggerIndent StepIndent{ Log };

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG() and TS.verifyLeafs());

  // Initialize a vector of nodes before iterating.
  // The algorithm iterates over all nodes in the graph, but it can end merging
  // the current node (and a bunch of others) with another one, and that would
  // invalidate the iterators if we iterate on llvm::nodes(&TS) directly.
  std::vector<LTSN *> Nodes{ llvm::nodes(&TS).begin(), llvm::nodes(&TS).end() };
  std::unordered_set<LTSN *> Erased;

  while (not Nodes.empty()) {
    LTSN *Node = Nodes.back();
    Nodes.pop_back();

    revng_log(Log, "Analyzing Node: " << Node->ID);
    LoggerIndent Indent{ Log };

    if (Erased.contains(Node)) {
      revng_log(Log, "merged by a previous iteration");
      continue;
    }

    if (isInstanceLeaf(Node)) {
      revng_log(Log, "no instance children");
      continue;
    }

    llvm::EquivalenceClasses<LTSN *> ToMerge;

    auto ChildEnd = Node->Successors.end();
    for (auto AChildIt = Node->Successors.begin(); AChildIt != ChildEnd;
         ++AChildIt) {

      const auto &AEdge = *AChildIt;
      if (not isInstanceEdge(AEdge))
        continue;

      const auto &[APointer, ATag] = AEdge;
      if (not hasOutgoingPointerEdge(APointer))
        continue;
      revng_assert(isWellFormedPointer(APointer));

      for (auto BChildIt = std::next(AChildIt); BChildIt != ChildEnd;
           ++BChildIt) {

        const auto &BEdge = *BChildIt;
        const auto &[BPointer, BTag] = BEdge;
        if (ATag != BTag)
          continue;

        if (not hasOutgoingPointerEdge(BPointer))
          continue;
        revng_assert(isWellFormedPointer(BPointer));

        // Here we're sure that A and B are connected to Node with the same kind
        // of instance edge. And that they are both pointer nodes.

        revng_log(Log,
                  "has a pair of instance children at the same offset that are "
                  "pointer nodes:");
        revng_log(Log, "A: " << APointer->ID << ", B:" << BPointer->ID);

        revng_assert(APointer->Successors.size() == 1,
                     std::to_string(APointer->ID).c_str());
        revng_assert(BPointer->Successors.size() == 1,
                     std::to_string(BPointer->ID).c_str());
        ToMerge.unionSets(APointer, BPointer);
      }
    }

    if (not ToMerge.empty()) {
      revng_log(Log, "Merging children");
      LoggerIndent MoreIndent{ Log };

      // Iterate over all of the equivalence sets.
      for (auto I = ToMerge.begin(), E = ToMerge.end(); I != E; ++I) {
        // Ignore non-leader sets.
        if (not I->isLeader())
          continue;

        // Loop over members in this set to select the node that we want to
        // merge the others into.
        const auto NotErased =
          [&Erased = std::as_const(Erased)](LayoutTypeSystemNode *N) {
            return not Erased.contains(N)
                   and not Erased.contains(getPointee(N));
          };
        using llvm::make_filter_range;
        auto
          Pointers = make_filter_range(llvm::make_range(ToMerge.member_begin(I),
                                                        ToMerge.member_end()),
                                       NotErased);

        if (Log.isEnabled()) {
          revng_log(Log, "Preparing to merge pointees:");
          LoggerIndent EvenMoreIndent{ Log };
          for (LayoutTypeSystemNode *N : Pointers)
            revng_log(Log,
                      N->ID << " with pointee " << getPointee(N)->ID
                            << " (size: " << getPointee(N)->Size << ")");
        }

        llvm::SmallSetVector<LTSN *, 8> UniquedScalars;
        llvm::SmallPtrSet<LTSN *, 8> PointersToScalars;
        llvm::SmallSetVector<LTSN *, 8> UniquedAggregates;
        llvm::SmallPtrSet<LTSN *, 8> PointersToAggregates;
        llvm::SmallSetVector<LTSN *, 8> UniquedFromModel;
        llvm::SmallPtrSet<LTSN *, 8> PointersToFromModel;

        for (LayoutTypeSystemNode *Pointer : Pointers) {
          revng_assert(not Erased.contains(Pointer));
          LTSN *Pointee = getPointee(Pointer);
          revng_assert(not Erased.contains(Pointee));

          if (Pointee->NonScalar) {
            PointersToFromModel.insert(Pointer);
            UniquedFromModel.insert(Pointee);
          } else if (Pointee->Successors.empty()
                     or hasOutgoingPointerEdge(Pointee)) {
            revng_assert(not hasOutgoingPointerEdge(Pointee)
                         or isWellFormedPointer(Pointee));
            PointersToScalars.insert(Pointer);
            UniquedScalars.insert(Pointee);
          } else {
            PointersToAggregates.insert(Pointer);
            UniquedAggregates.insert(Pointee);
          }
        }

        // Sort scalars and aggregates so that the first is the node with the
        // lowerst ID among the nodes with largest size.

        llvm::SmallVector<LTSN *> Scalars = UniquedScalars.takeVector();
        llvm::SmallVector<LTSN *> Aggregates = UniquedAggregates.takeVector();
        llvm::SmallVector<LTSN *> FromModel = UniquedAggregates.takeVector();

        const auto Ordering = [](const LTSN *LHS, const LTSN *RHS) {
          auto LSize = LHS->Size;
          auto RSize = RHS->Size;
          if (LSize > RSize)
            return true;

          if (LSize == RSize)
            return LHS->ID < RHS->ID;

          return false;
        };

        llvm::sort(Scalars, Ordering);
        llvm::sort(Aggregates, Ordering);
        llvm::sort(FromModel, Ordering);

        // Merge all the scalars together.
        LTSN *MergedScalar = nullptr;
        if (not Scalars.empty()) {
          if (Log.isEnabled()) {
            revng_log(Log, "merging Scalars:");
            LoggerIndent MoreMoreIndent{ Log };
            for (const LTSN *N : Scalars)
              revng_log(Log, N->ID);
          }
          TS.mergeNodes(Scalars);
          Erased.insert(std::next(Scalars.begin()), Scalars.end());
          MergedScalar = Scalars.front();

          // Check if we merged more than one scalar that also was a pointer.
          // In that case we have to create a new union of their pointees,
          // enqueue it for further analysis
          llvm::SmallVector<LTSN::NeighborIterator> PointerEdges;
          {
            LTSN::NeighborIterator ChildIt = MergedScalar->Successors.begin();
            LTSN::NeighborIterator ChildEnd = MergedScalar->Successors.end();
            for (; ChildIt != ChildEnd; ++ChildIt)
              if (isPointerEdge(*ChildIt))
                PointerEdges.push_back(ChildIt);

            revng_assert(PointerEdges.empty()
                         or MergedScalar->Size == PointerSize);
          }

          if (PointerEdges.size() > 1) {
            revng_log(Log,
                      "Merged scalar is a union of pointers: "
                        << MergedScalar->ID);
            for (LTSN::NeighborIterator &PointerEdgeIt : PointerEdges) {
              LTSN *NewPointer = TS.createArtificialLayoutType();
              NewPointer->Size = PointerSize;
              TS.moveEdgeSource(MergedScalar, NewPointer, PointerEdgeIt, 0);
              TS.addInstanceLink(MergedScalar,
                                 NewPointer,
                                 OffsetExpression{ 0 });
            }
            Nodes.push_back(MergedScalar);
          }
        }

        LTSN *MergedAggregate = nullptr;
        if (not Aggregates.empty()) {
          if (Log.isEnabled()) {
            revng_log(Log, "merging Aggregates:");
            LoggerIndent MoreMoreIndent{ Log };
            for (const LTSN *N : Aggregates)
              revng_log(Log, N->ID);
          }
          MergedAggregate = Aggregates.front();
          TS.mergeNodes(Aggregates);
          Erased.insert(std::next(Aggregates.begin()), Aggregates.end());
        }

        if (MergedAggregate and not FromModel.empty()) {
          size_t AggregateSize = MergedAggregate->Size;
          auto It = llvm::find_if(FromModel, [AggregateSize](LTSN *N) {
            return N->Size < AggregateSize;
          });
          auto SmallFromModel = llvm::make_range(It, FromModel.end());

          for (LTSN *AggregateFromModel : SmallFromModel) {
            revng_assert(AggregateFromModel->Size < AggregateSize);
            // We want to move the pointers that point to AggregateFromModel,
            // and make it point to MergedAggregate.
            const auto PredecessorPointsToAggregate =
              [&PointersToFromModel](const auto &Pair) {
                return isPointerEdge(Pair)
                       and PointersToFromModel.contains(Pair.first);
              };
            auto InverseEdgeIt = llvm::find_if(AggregateFromModel->Predecessors,
                                               PredecessorPointsToAggregate);
            TS.moveEdgeTarget(AggregateFromModel,
                              MergedAggregate,
                              InverseEdgeIt,
                              0);

            // We want to put an instance link at offset 0 from MergedAggregate
            // to each small AggregateFromModel.
            TS.addInstanceLink(MergedAggregate,
                               AggregateFromModel,
                               OffsetExpression{ 0 });
          }

          if (1 == std::distance(FromModel.begin(), It)) {
            LTSN *UniqueFromModelToPreserve = *FromModel.begin();

            revng_log(Log,
                      "UniqueFromModelToPreserve: "
                        << (UniqueFromModelToPreserve ?
                              std::to_string(UniqueFromModelToPreserve->ID) :
                              "none"));
            TS.mergeNodes({ UniqueFromModelToPreserve, MergedAggregate });
            Erased.insert(MergedAggregate);
            MergedAggregate = UniqueFromModelToPreserve;
          }
        }

        if (MergedAggregate and MergedScalar) {

          // First, we want all pointers that point to MergedScalar to actually
          // start pointing to MergedAggregate.
          for (LTSN *Pointer : PointersToScalars) {
            const auto &[Pointee, PointerTag] = *Pointer->Successors.begin();
            auto InverseEdgeIt = Pointee->Predecessors.find({ Pointer,
                                                              PointerTag });
            TS.moveEdgeTarget(Pointee, MergedAggregate, InverseEdgeIt, 0);
          }

          // Second, we want to inject an instance of MergedScalar at offset 0
          // inside MergedAggregate.
          // If MergedAggregate is larger than MergedScalar we're fine.
          if (MergedAggregate->Size >= MergedScalar->Size) {
            TS.addInstanceLink(MergedAggregate,
                               MergedScalar,
                               OffsetExpression{ 0 });
          } else if (not MergedAggregate->NonScalar) {
            MergedAggregate->Size = MergedScalar->Size;
            TS.addInstanceLink(MergedAggregate,
                               MergedScalar,
                               OffsetExpression{ 0 });
          } else {
            revng_abort();
          }
        }
      }
    }
  }

  return Changed;
}

} // end namespace dla
