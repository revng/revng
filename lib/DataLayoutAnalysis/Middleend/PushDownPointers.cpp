//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"

#include "revng/ADT/FilteredGraphTraits.h"

#include "DLAStep.h"

namespace dla {

using GT = llvm::GraphTraits<LayoutTypeSystemNode *>;

static bool isInstanceAtOffset0(const GT::EdgeRef &E) {
  if (not isInstanceEdge(E))
    return false;

  return not E.second->getOffsetExpr().Offset;
}

using Instance0Graph = EdgeFilteredGraph<dla::LayoutTypeSystemNode *,
                                         isInstanceEdge>;

using CInstance0Graph = EdgeFilteredGraph<const dla::LayoutTypeSystemNode *,
                                          isInstanceAtOffset0>;

using Instance0Inverse = llvm::Inverse<Instance0Graph>;

using CInstance0Inverse = llvm::Inverse<CInstance0Graph>;

using DFVisitSet = llvm::df_iterator_default_set<const LayoutTypeSystemNode *,
                                                 8>;

static DFVisitSet getGrandParentsAtOffset0(const LayoutTypeSystemNode *N) {

  DFVisitSet Result;
  for ([[maybe_unused]] const auto *_ :
       llvm::depth_first_ext(CInstance0Inverse(N), Result))
    ;
  return Result;
}

static DFVisitSet getChildrenAtOffset0(const LayoutTypeSystemNode *Parent) {
  DFVisitSet Children;
  const auto
    &Instance0SuccessorsRange = llvm::make_filter_range(Parent->Successors,
                                                        isInstanceAtOffset0);
  for (const LayoutTypeSystemNode *Child :
       llvm::make_first_range(Instance0SuccessorsRange))
    Children.insert(Child);
  return Children;
}

static uint64_t getNumChildrenAtOffset0(const LayoutTypeSystemNode *Parent) {
  return getChildrenAtOffset0(Parent).size();
}

bool PushDownPointers::runOnTypeSystem(LayoutTypeSystem &TS) {

  bool Changed = false;

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  for (LayoutTypeSystemNode *Parent : llvm::nodes(&TS)) {

    if (isInstanceLeaf(Parent))
      continue;

    revng_assert(not isPointerNode(Parent));

    DFVisitSet GrandParentsAtOffset0 = getGrandParentsAtOffset0(Parent);
    llvm::SmallVector<LayoutTypeSystemNode *, 8> PointersToGrandParents;

    {
      llvm::SmallPtrSet<LayoutTypeSystemNode *, 8> Visited;
      for (LayoutTypeSystemNode *Child :
           llvm::make_first_range(Parent->Successors)) {
        for (LayoutTypeSystemNode *Pointer :
             llvm::post_order_ext(Child, Visited)) {
          if (not isPointerNode(Pointer))
            continue;
          revng_assert(Pointer->Successors.size() == 1);
          LayoutTypeSystemNode *Pointee = Pointer->Successors.begin()->first;
          if (Pointee != Parent and GrandParentsAtOffset0.count(Pointee))
            PointersToGrandParents.push_back(Pointer);
        }
      }
    }

    for (LayoutTypeSystemNode *Pointer : PointersToGrandParents) {
      revng_assert(isPointerNode(Pointer) and Pointer->Successors.size() == 1);
      auto PointerEdgeIt = Pointer->Successors.begin();
      LayoutTypeSystemNode *Pointee = PointerEdgeIt->first;
      LayoutTypeSystemNode *NewPointee = nullptr;
      DFVisitSet Visited = getChildrenAtOffset0(Parent);
      for (LayoutTypeSystemNode *N :
           llvm::depth_first_ext(Instance0Graph(Pointee), Visited)) {
        if (getNumChildrenAtOffset0(N) > 1) {
          NewPointee = N;
          break;
        }
      }

      if (not NewPointee)
        NewPointee = Parent;

      TS.eraseEdge(Pointer, PointerEdgeIt);
      TS.addPointerLink(Pointer, NewPointee);
      Changed = true;
    }
  }

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return Changed;
}

} // end namespace dla
