//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

#include "DLAHelpers.h"

#include "DLATypeSystem.h"

using std::conditional_t;

template<typename ValT>
using LLVMValueT = conditional_t<std::is_const_v<ValT>,
                                 const llvm::Value,
                                 llvm::Value>;

template<typename T>
std::enable_if_t<std::is_same_v<std::remove_const_t<T>, llvm::InsertValueInst>,
                 llvm::SmallVector<LLVMValueT<T> *, 2>>
getConstQualifiedInsertValueLeafOperands(T *Ins) {
  using ValueT = LLVMValueT<T>;
  llvm::SmallVector<ValueT *, 2> Results;
  llvm::SmallSet<unsigned, 2> FoundIds;
  auto *StructTy = llvm::cast<llvm::StructType>(Ins->getType());
  unsigned NumFields = StructTy->getNumElements();
  Results.resize(NumFields, nullptr);
  revng_assert(Ins->getNumUses() == 1
               and (isa<llvm::InsertValueInst>(Ins->use_begin()->getUser())
                    or isa<llvm::ReturnInst>(Ins->use_begin()->getUser())));
  while (1) {
    revng_assert(Ins->getNumIndices() == 1);
    unsigned FieldId = Ins->getIndices()[0];
    revng_assert(FieldId < NumFields);
    revng_assert(FoundIds.count(FieldId) == 0);
    FoundIds.insert(FieldId);
    ValueT *Op = Ins->getInsertedValueOperand();
    revng_assert(isa<llvm::IntegerType>(Op->getType())
                 or isa<llvm::PointerType>(Op->getType()));
    revng_assert(Results[FieldId] == nullptr);
    Results[FieldId] = Op;
    ValueT *Tmp = Ins->getAggregateOperand();
    Ins = llvm::dyn_cast<llvm::InsertValueInst>(Tmp);
    if (not Ins) {
      revng_assert(llvm::isa<llvm::UndefValue>(Tmp)
                   or llvm::isa<llvm::ConstantAggregate>(Tmp));
      break;
    }
  }
  return Results;
};

llvm::SmallVector<llvm::Value *, 2>
getInsertValueLeafOperands(llvm::InsertValueInst *Ins) {
  return getConstQualifiedInsertValueLeafOperands(Ins);
}

llvm::SmallVector<const llvm::Value *, 2>
getInsertValueLeafOperands(const llvm::InsertValueInst *Ins) {
  return getConstQualifiedInsertValueLeafOperands(Ins);
}

template<typename T>
std::enable_if_t<std::is_same_v<std::remove_const_t<T>, llvm::CallInst>,
                 llvm::SmallVector<LLVMValueT<T> *, 2>>
getConstQualifiedExtractedValuesFromCall(T *Call) {
  using ValueT = LLVMValueT<T>;
  llvm::SmallVector<ValueT *, 2> Results;
  llvm::SmallSet<unsigned, 2> FoundIds;
  auto *StructTy = llvm::cast<llvm::StructType>(Call->getType());
  unsigned NumFields = StructTy->getNumElements();
  Results.resize(NumFields, nullptr);
  revng_assert(Call->getNumUses() <= NumFields);
  for (auto *Extract : Call->users()) {
    auto *E = cast<llvm::ExtractValueInst>(Extract);
    revng_assert(E->getNumIndices() == 1);
    unsigned FieldId = E->getIndices()[0];
    revng_assert(FieldId < NumFields);
    revng_assert(FoundIds.count(FieldId) == 0);
    FoundIds.insert(FieldId);
    revng_assert(isa<llvm::IntegerType>(E->getType())
                 or isa<llvm::PointerType>(E->getType()));
    revng_assert(Results[FieldId] == nullptr);
    Results[FieldId] = E;
  }
  return Results;
};

llvm::SmallVector<llvm::Value *, 2>
getExtractedValuesFromCall(llvm::CallInst *Call) {
  return getConstQualifiedExtractedValuesFromCall(Call);
}

llvm::SmallVector<const llvm::Value *, 2>
getExtractedValuesFromCall(const llvm::CallInst *Call) {
  return getConstQualifiedExtractedValuesFromCall(Call);
}

uint64_t getLoadStoreSizeFromPtrOpUse(const dla::LayoutTypeSystem &TS,
                                      const llvm::Use *U) {
  llvm::Value *AddrOperand = U->get();
  auto *PtrTy = cast<llvm::PointerType>(AddrOperand->getType());
  llvm::Type *AccessedT = PtrTy->getElementType();
  const llvm::DataLayout &DL = TS.getModule().getDataLayout();
  return DL.getTypeAllocSize(AccessedT);
};

static Logger<> Log("dla-instance-inheritance-loops");

namespace dla {

using LTSN = LayoutTypeSystemNode;
using GraphNodeT = LTSN *;
using InheritanceNodeT = EdgeFilteredGraph<GraphNodeT, isInheritanceEdge>;
using CGraphNodeT = const LTSN *;
using CInheritanceNodeT = EdgeFilteredGraph<CGraphNodeT, isInheritanceEdge>;

static bool
isInheritanceOrInstanceEdge(const llvm::GraphTraits<LTSN *>::EdgeRef &E) {
  return isInheritanceEdge(E) or isInstanceEdge(E);
}

using MixedNodeT = EdgeFilteredGraph<GraphNodeT, isInheritanceOrInstanceEdge>;
using MixedGT = llvm::GraphTraits<MixedNodeT>;

bool removeInstanceBackedgesFromInheritanceLoops(LayoutTypeSystem &TS) {
  bool Changed = false;
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyInheritanceDAG());
  if (Log.isEnabled())
    TS.dumpDotOnFile("before-remove-instance-inheritance-loops.dot");

  revng_log(Log, "Removing Instance Backedges From Inheritance Loops");

  const auto HasNoInheritanceEdge = [](const LTSN *Node) {
    return isInheritanceRoot(Node) and isInheritanceLeaf(Node);
  };

  // Color all the nodes, except those that have no incoming nor outgoing
  // inheritance edges.
  // The goal is to identify the subsets of nodes that are connected by means of
  // inheritance edges, meaning that they have some form of inheritance
  // relationship (even if not direct).
  // In this way we divide the graph in subgraphs, such that for each pair of
  // nodes P and Q with (P != Q) in the same sugraphs (i.e. with the same
  // color), either P inherits from Q, or Q inherits from P (even if not
  // directly). Each of this subgraphs is called "inheritance component".
  // The idea is that inheritance edges are more meaningful than instance edges,
  // so we don't want to remove any of them, but we need to identify instance
  // edges that create loops across multiple inheritance components, and cut
  // them.
  std::map<const LTSN *, unsigned> NodeColors;
  {
    // Holds a set of nodes.
    using NodeSet = llvm::df_iterator_default_set<const LTSN *, 16>;

    // Map colors to set of nodes with that color.
    std::map<unsigned, NodeSet> ColorToNodes;
    unsigned NewColor = 0UL;

    for (const LTSN *Root : llvm::nodes(&TS)) {
      revng_assert(Root != nullptr);
      // Skip nodes that have no incoming or outgoing inheritance edges.
      if (HasNoInheritanceEdge(Root))
        continue;
      // Start visiting only from inheritance roots.
      if (not isInheritanceRoot(Root))
        continue;

      // Depth first visit across inheritance edges.
      llvm::df_iterator_default_set<const LayoutTypeSystemNode *, 16> Visited;
      // Tracks the set of colors we found during this visit.
      llvm::SmallSet<unsigned, 16> FoundColors;
      for (auto *N : llvm::depth_first_ext(CInheritanceNodeT(Root), Visited)) {
        // If N is colored, we have already visited it starting from another
        // Root. We add it to the FoundColors and mark its inheritance children
        // as visited, so that they are skipped in the depth first visit.
        if (auto NodeColorIt = NodeColors.find(N);
            NodeColorIt != NodeColors.end()) {
          unsigned Color = NodeColorIt->second;
          FoundColors.insert(Color);
          for (const LTSN *Child : llvm::children<CInheritanceNodeT>(N))
            Visited.insert(Child);
        }
      }

      // Add the visited nodes to the ColorToNodesMap, with a new color.
      auto It = ColorToNodes.insert({ NewColor, std::move(Visited) }).first;
      // If we encountered other colors during the visit, all the merged colors
      // need to be merged into the new color.
      if (not FoundColors.empty()) {
        llvm::SmallVector<decltype(ColorToNodes)::iterator, 8> OldToErase;
        // Merge all the sets of nodes with the colors we found with the new
        // set of nodes with the new color.
        for (unsigned OldColor : FoundColors) {
          auto ColorToNodesIt = ColorToNodes.find(OldColor);
          revng_assert(ColorToNodesIt != ColorToNodes.end());
          auto &OldColoredNodes = ColorToNodesIt->second;
          It->second.insert(OldColoredNodes.begin(), OldColoredNodes.end());
          // Mark this iterator as OldToErase, because after we're done merging
          // the old color sets need to be dropped.
          OldToErase.push_back(ColorToNodesIt);
        }

        // Drop the set of nodes with old colors.
        for (auto &ColorToNodesIt : OldToErase)
          ColorToNodes.erase(ColorToNodesIt);
      }

      // Set the proper color to all the newly found nodes.
      for (auto *Node : It->second)
        NodeColors[Node] = NewColor;

      ++NewColor;
    }
  }

  // Here all the nodes are colored.
  // Each inheritance component has a different color, while nodes that have no
  // incoming or outgoing inheritance edges do not have a color.

  for (const auto &Root : llvm::nodes(&TS)) {
    revng_assert(Root != nullptr);
    // We start from inheritance roots and look if we find an SCC with mixed
    // edges (instance and inheritance).
    if (HasNoInheritanceEdge(Root))
      continue;

    if (not isInheritanceRoot(Root))
      continue;

    revng_log(Log,
              "# Looking for mixed instance inheritance loops from: "
                << Root->ID);

    struct EdgeInfo {
      LTSN *Src;
      LTSN *Tgt;
      const TypeLinkTag *Tag;
      // Comparison operators to use in set
      bool operator<(const EdgeInfo &O) const {
        using std::make_tuple;
        return make_tuple(Src, Tgt, Tag) < make_tuple(O.Src, O.Tgt, O.Tag);
      }
      bool operator==(const EdgeInfo &O) const {
        using std::make_tuple;
        return make_tuple(Src, Tgt, Tag) == make_tuple(O.Src, O.Tgt, O.Tag);
      }
    };

    llvm::SmallPtrSet<const LayoutTypeSystemNode *, 16> Visited;
    llvm::SmallPtrSet<const LayoutTypeSystemNode *, 16> InStack;

    struct StackEntry {
      LayoutTypeSystemNode *Node;
      unsigned Color;
      MixedGT::ChildEdgeIteratorType NextToVisitIt;
    };
    std::vector<StackEntry> VisitStack;

    const auto TryPush = [&](LTSN *N, unsigned Color) {
      revng_log(Log, "--* try_push(" << N->ID << ')');
      bool NewVisit = Visited.insert(N).second;
      if (NewVisit) {
        revng_log(Log, "    color: " << Color);
        revng_assert(Color != std::numeric_limits<unsigned>::max());

        VisitStack.push_back({ N, Color, MixedGT::child_edge_begin(N) });
        InStack.insert(N);
        revng_assert(InStack.size() == VisitStack.size());
        revng_log(Log, "--> pushed!");
      } else {
        revng_log(Log, "--| already visited!");
      }
      return NewVisit;
    };

    const auto Pop = [&VisitStack, &InStack]() {
      revng_log(Log, "<-- pop(" << VisitStack.back().Node->ID << ')');
      InStack.erase(VisitStack.back().Node);
      VisitStack.pop_back();
      revng_assert(InStack.size() == VisitStack.size());
    };

    llvm::SmallSet<EdgeInfo, 8> ToRemove;
    llvm::SmallVector<EdgeInfo, 8> CrossColorEdges;

    TryPush(Root, NodeColors.at(Root));
    while (not VisitStack.empty()) {
      StackEntry &Top = VisitStack.back();

      unsigned TopColor = Top.Color;
      LTSN *TopNode = Top.Node;
      MixedGT::ChildEdgeIteratorType &NextEdgeToVisit = Top.NextToVisitIt;

      revng_log(Log,
                "## Stack top is: " << TopNode->ID
                                    << "\n          color: " << TopColor);

      bool StartNew = false;
      while (NextEdgeToVisit != MixedGT::child_edge_end(TopNode)
             and not StartNew) {

        LTSN *NextChild = NextEdgeToVisit->first;
        const TypeLinkTag *NextTag = NextEdgeToVisit->second;
        EdgeInfo E = { TopNode, NextChild, NextTag };

        revng_log(Log, "### Next child:: " << NextChild->ID);

        // Check if the next children is colored.
        // If it's not, leave the same color of the top of the stack, so that we
        // can identify the first edge that closes the crossing from one
        // inheritance component to another.
        unsigned NextColor = TopColor;
        if (auto ColorsIt = NodeColors.find(NextChild);
            ColorsIt != NodeColors.end()) {
          revng_log(Log, "Colored");
          NextColor = ColorsIt->second;
          if (NextColor != TopColor) {
            revng_log(Log,
                      "Push Cross-Color Edge " << TopNode->ID << " -> "
                                               << NextChild->ID);
            revng_assert(E.Tag->getKind() == TypeLinkTag::LK_Instance);
            CrossColorEdges.push_back(std::move(E));
          }
        }

        ++NextEdgeToVisit;
        StartNew = TryPush(NextChild, NextColor);

        if (not StartNew) {

          // We haven't pushed, either because NextChild is on the stack, or
          // because it was visited before.
          if (InStack.count(NextChild)) {

            // If it's on the stack, we're closing a loop.
            // Add all the cross color edges to the edges ToRemove.
            revng_log(Log, "Closes Loop");
            if (Log.isEnabled()) {
              for (EdgeInfo &E : CrossColorEdges) {
                revng_log(Log,
                          "Is to remove: " << E.Src->ID << " -> " << E.Tgt->ID);
              }
            }
            ToRemove.insert(CrossColorEdges.begin(), CrossColorEdges.end());

            // This an optimization.
            // All the CrossColorEdges have just been added to the edges
            // ToRemove, so there's no point keeping them also in
            // CrossColorEdges, and possibly trying to insert them again later.
            // We can drop all of them here.
            CrossColorEdges.clear();

            if (NextColor == TopColor
                and E.Tag->getKind() == TypeLinkTag::LK_Instance) {
              // This means that the edge E we tried to push on the stack is an
              // instance edge closing a loop.
              // The loop can be either entirely composed of instance edges, or
              // can be composed by some inheritance edges belonging to a single
              // inheritance components with a retreating instance edge that
              // targets the same inheritance component.
              // In all these cases, the retreating edge is an instance link,
              // and we must remove it.
              ToRemove.insert(std::move(E));
            }
          }

          if (NextColor != TopColor and not CrossColorEdges.empty()) {
            EdgeInfo E = CrossColorEdges.pop_back_val();
            revng_log(Log,
                      "Pop Cross-Color Edge " << E.Src->ID << " -> "
                                              << E.Tgt->ID);
          }
        }
      }

      if (StartNew) {
        // We exited the push loop with a TryPush succeeding, so we need to look
        // at the new child freshly pushed on the stack.
        continue;
      }

      revng_log(Log, "## Completed : " << TopNode->ID);

      Pop();

      if (not VisitStack.empty() and not CrossColorEdges.empty()
          and TopColor != VisitStack.back().Color) {
        // We are popping back a cross-color edge. Remove it.
        EdgeInfo E = CrossColorEdges.pop_back_val();
        revng_log(Log,
                  "Pop Cross-Color Edge " << E.Src->ID << " -> " << E.Tgt->ID);
      }
    }

    // Actually remove the edges
    for (auto &[Pred, Child, T] : ToRemove) {
      using Edge = LTSN::NeighborsSet::value_type;
      revng_log(Log,
                "# Removing instance edge: " << Pred->ID << " -> "
                                             << Child->ID);
      revng_assert(T->getKind() == TypeLinkTag::LK_Instance);
      Edge ChildToPred = std::make_pair(Pred, T);
      bool Erased = Child->Predecessors.erase(ChildToPred);
      revng_assert(Erased);
      Edge PredToChild = std::make_pair(Child, T);
      Erased = Pred->Successors.erase(PredToChild);
      revng_assert(Erased);
      Changed = true;
    }
  }

  return Changed;
} // namespace dla

} // end namespace dla
