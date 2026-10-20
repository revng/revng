//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_set>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CFG.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLAStep.h"

using namespace llvm;

static Logger Log("dla-merge-pointees-of-ptr-union");

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

/// \return the number of pointer edges on the chain that starts at \p N
///
/// A node with no outgoing pointer edge has length 0. A well formed pointer
/// has a single successor, so the chain it starts is unique, and pointer edges
/// form a DAG, so walking it terminates.
static unsigned pointerChainLength(const LTSN *N) {
  unsigned Length = 0;
  while (hasOutgoingPointerEdge(N)) {
    revng_assert(N->Successors.size() == 1);
    N = N->Successors.begin()->first;
    ++Length;
  }
  return Length;
}

/// \return the length of the chain of pointer edges below the pointee of \p P
static unsigned pointeeChainLength(LTSN *P) {
  return pointerChainLength(getPointee(P));
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

    llvm::EquivalenceClasses<LTSN *> PointersToMerge;

    // The pointers that could be merged, by the tag of the instance edge that
    // reaches them and by the length of the chain of pointer edges below their
    // pointee. A class of a single pointer is not in `PointersToMerge`, which
    // only holds what has been unioned, so they are collected here too.
    std::map<const TypeLinkTag *, std::map<unsigned, llvm::SmallVector<LTSN *>>>
      CandidatesByTag;

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

      const unsigned ALength = pointeeChainLength(APointer);
      CandidatesByTag[ATag][ALength].push_back(APointer);

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
        revng_log(Log, "ATag: " << *ATag << ", BTag: " << *BTag);

        revng_assert(APointer->Successors.size() == 1,
                     std::to_string(APointer->ID).c_str());
        revng_assert(BPointer->Successors.size() == 1,
                     std::to_string(BPointer->ID).c_str());
        // Only put in the same class pointers whose pointees have chains of
        // pointer edges of the same length. Reaching one pointee from another
        // takes at least one pointer edge, which makes the chain of the one
        // that reaches strictly longer, so a class of equal lengths can be
        // merged without ever closing a loop of pointer edges. Equality is
        // transitive, so the classes stay uniform however they are built up.
        if (ALength != pointeeChainLength(BPointer)) {
          revng_log(Log, "pointee chains have different lengths: not merging");
          continue;
        }

        PointersToMerge.unionSets(APointer, BPointer);
      }
    }

    // Phase 1 only put in one class pointers whose pointees have chains of
    // equal length, so merging a class never closes a loop of pointer edges.
    // The pointees of the class of length 0 are not pointers, so they reach
    // nothing, and merging them into a deeper class of the same tag is safe
    // unless that class reaches them. Join the shallowest one that does not.
    for (auto &[Tag, ByLength] : CandidatesByTag) {
      auto ZeroIt = ByLength.find(0);
      if (ZeroIt == ByLength.end() or ByLength.size() < 2)
        continue;

      const llvm::SmallVector<LTSN *> &Zero = ZeroIt->second;

      // Everything that reaches a pointee of the class of length 0, walked
      // once, against the pointer edges.
      using CPointerT = EdgeFilteredGraph<const LTSN *, isPointerEdge>;
      llvm::df_iterator_default_set<const LTSN *> Reaching;
      for (LTSN *Pointer : Zero) {
        const LTSN *Pointee = getPointee(Pointer);
        for (const LTSN *Upstream :
             llvm::inverse_depth_first_ext(CPointerT(Pointee), Reaching))
          revng_log(Log,
                    "reaches a pointee of the class of length 0: "
                      << Upstream->ID);
      }

      // The map is sorted, so the shallowest classes come first.
      for (auto &[Length, Pointers] : ByLength) {
        if (Length == 0)
          continue;

        const auto ReachesZeroClass = [&Reaching](LTSN *Pointer) {
          return Reaching.contains(getPointee(Pointer));
        };
        if (llvm::any_of(Pointers, ReachesZeroClass)) {
          revng_log(Log,
                    "the class of length " << Length
                                           << " reaches the pointees of the "
                                              "class of length 0");
          continue;
        }

        revng_log(Log,
                  "merging the class of length 0 into the one of length "
                    << Length);
        for (LTSN *ZeroPointer : Zero)
          PointersToMerge.unionSets(ZeroPointer, Pointers.front());
        break;
      }
    }

    if (not PointersToMerge.empty()) {
      revng_log(Log, "Merging children");
      LoggerIndent MoreIndent{ Log };

      // Iterate over all of the equivalence sets.
      for (auto I = PointersToMerge.begin(), E = PointersToMerge.end(); I != E;
           ++I) {
        // Ignore non-leader sets.
        if (not I->isLeader())
          continue;

        // Loop over members in this set to select the node that we want to
        // merge the others into.
        const auto NotErasedPointer =
          [&Erased = std::as_const(Erased)](LayoutTypeSystemNode *N) {
            revng_log(Log, "NotErased ID: " << N->ID);
            LoggerIndent Indent{ Log };

            // Skip erased nodes.
            if (Erased.contains(N)) {
              revng_log(Log, "Erased");
              return false;
            }

            // Skip non-pointer nodes.
            // Nodes in PointersToMerge, being scalar, can be pointed to by
            // other pointers whose pointees are also scalars to merge.
            // This means that in some cases a pointer node can be merged with
            // its scalar pointee, turning it first into a scalar with an
            // outgoing pointer edge to self, and then into a non-pointer scalar
            // (because self edges are cancelled).
            // In those cases, we might end up with a node that was in
            // PointersToMerge being turned into a non-pointer.
            // If that happens we just ignore it, because it's not a pointer
            // anymore, and we just look at others.
            if (not hasOutgoingPointerEdge(N)) {
              revng_log(Log, "NOT hasOutgoingPointerEdge");
              return false;
            }

            // If the pointee was erased, the outgoing pointer edge from N would
            // have been erased too, and we would have fallen in the check
            // above.
            revng_assert(not Erased.contains(getPointee(N)));
            return true;
          };
        using llvm::make_filter_range;
        auto Pointers = make_filter_range(llvm::make_range(PointersToMerge
                                                             .member_begin(I),
                                                           PointersToMerge
                                                             .member_end()),
                                          NotErasedPointer);

        if (Log.isEnabled()) {
          revng_log(Log, "Preparing to merge pointees:");
          LoggerIndent EvenMoreIndent{ Log };
          for (LayoutTypeSystemNode *N : Pointers) {
            const LTSN *Pointee = getPointee(N);
            std::string ScalarString = Pointee->NonScalar ? "Aggregate" :
                                                            "Scalar";
            revng_log(Log,
                      N->ID << " with pointee " << Pointee->ID << " ("
                            << ScalarString << ", size: " << getPointee(N)->Size
                            << ")");
          }
          revng_log(Log, "----");
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
        llvm::SmallVector<LTSN *> FromModel = UniquedFromModel.takeVector();

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

          revng_log(Log, "MergedScalar: " << MergedScalar->ID);

          // A node with an outgoing pointer edge is a pointer, so it is
          // pointer sized and that edge is the only one it has. Merging can
          // break both of those: more than one of the merged scalars can be a
          // pointer, and a pointer can be merged into a scalar larger than
          // itself, since what survives a merge is the largest of the nodes.
          llvm::SmallVector<LTSN::NeighborIterator> PointerEdges;
          {
            LTSN::NeighborIterator ChildIt = MergedScalar->Successors.begin();
            LTSN::NeighborIterator ChildEnd = MergedScalar->Successors.end();
            for (; ChildIt != ChildEnd; ++ChildIt)
              if (isPointerEdge(*ChildIt))
                PointerEdges.push_back(ChildIt);
          }

          // Whatever carries a pointer edge is at least as large as a pointer,
          // and the merged scalar is at least as large as any of them.
          revng_assert(PointerEdges.empty()
                       or MergedScalar->Size >= PointerSize);

          // Give each pointer edge a pointer sized node of its own, reachable
          // from the merged scalar at offset 0, unless the merged scalar is a
          // well formed pointer already, which is the case when it is pointer
          // sized and there is a single edge to hold.
          bool IsWellFormed = PointerEdges.size() == 1
                              and MergedScalar->Size == PointerSize;
          if (not PointerEdges.empty() and not IsWellFormed) {
            revng_log(Log,
                      "Merged scalar cannot hold its pointer edges: "
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
          revng_log(Log, "MergedAggregate: " << MergedAggregate->ID);
          TS.mergeNodes(Aggregates);
          Erased.insert(std::next(Aggregates.begin()), Aggregates.end());
        }

        if (MergedAggregate and not FromModel.empty()) {
          revng_log(Log, "merged aggregate from model");
          size_t AggregateSize = MergedAggregate->Size;
          auto It = llvm::find_if(FromModel, [AggregateSize](LTSN *N) {
            return N->Size < AggregateSize;
          });
          auto SmallFromModel = llvm::make_range(It, FromModel.end());

          for (LTSN *AggregateFromModel : SmallFromModel) {
            revng_assert(AggregateFromModel->Size < AggregateSize);
            revng_log(Log,
                      "AggregateFromModel: " << AggregateFromModel->ID
                                             << " Size: "
                                             << AggregateFromModel->Size);
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
          revng_log(Log, "MergedAggregate and MergedScalar");

          // First, we want all pointers that point to MergedScalar to actually
          // start pointing to MergedAggregate.
          for (LTSN *Pointer : PointersToScalars) {
            revng_log(Log, "PointerToScalar: " << Pointer->ID);
            if (Erased.contains(Pointer)) {
              // This can happen if one of the PointersToScalars was also itself
              // a pointee of another pointer to scalar.
              revng_log(Log, "was erased");
              continue;
            }

            if (Pointer == MergedScalar
                and not hasOutgoingPointerEdge(Pointer)) {
              revng_log(Log,
                        "Pointer was merged with its pointee scalar, and is no "
                        "longer a pointer");
              continue;
            }

            const auto &[Pointee, PointerTag] = *Pointer->Successors.begin();
            revng_log(Log, "Pointee: " << Pointee->ID);
            auto InverseEdgeIt = Pointee->Predecessors.find({ Pointer,
                                                              PointerTag });
            TS.moveEdgeTarget(Pointee, MergedAggregate, InverseEdgeIt, 0);
          }
          revng_log(Log, "Fixed up PointersToScalars");

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
