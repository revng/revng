//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include <iterator>
#include <set>
#include <stack>
#include <string>

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/ADT/SmallMap.h"
#include "revng/Support/Debug.h"

#include "DLAStep.h"
#include "DLATypeSystem.h"

static Logger<> Log("dla-remove-transitive-edges");

using namespace llvm;

namespace dla {

using LTSN = LayoutTypeSystemNode;
using InheritanceNodeT = EdgeFilteredGraph<LTSN *, isInheritanceEdge>;

class DFSStack {
public:
  struct StackEntry {
    LTSN *Node;
    llvm::SmallVector<LTSN *, 2> OrderedChildren;
    llvm::SmallVectorImpl<LTSN *>::size_type NextChild;
  };

private:
  llvm::SmallVector<StackEntry, 8> VisitStack;
  llvm::SmallPtrSet<LTSN *, 8> InStack;
  llvm::SmallPtrSet<LTSN *, 16> Visited;
  SmallMap<const LTSN *, unsigned, 16> PostOrder;

public:
  DFSStack(LTSN *Root) : VisitStack(), InStack(), Visited(), PostOrder() {
    unsigned O = 0U;
    for (LTSN *N : post_order(InheritanceNodeT(Root)))
      PostOrder[N] = O++;
  }

  bool tryPush(LTSN *N) {
    bool Inserted = Visited.insert(N).second;
    revng_log(Log, "--* try_push(" << N->ID << ')');
    if (Inserted) {
      // Get the children.
      llvm::SmallVector<LTSN *, 2> OrderedChildren;
      for (LTSN *C : children<InheritanceNodeT>(N))
        OrderedChildren.push_back(C);
      // Sort the children in reverse post order, so that we can traverse them
      // starting from those that are "closer to entry".
      const auto RPostOrderLess = [this](const LTSN *A, const LTSN *B) {
        return PostOrder.at(A) > PostOrder.at(B);
      };
      std::sort(OrderedChildren.begin(), OrderedChildren.end(), RPostOrderLess);

      // Push it on the stack
      VisitStack.push_back({ N, std::move(OrderedChildren), 0 });

      // Track it in the InStack set. This is necessary to be able to query
      // it, for detecting transitive edges.
      InStack.insert(N);
      revng_log(Log, "--> pushed!");
    } else {
      revng_log(Log, "--| already visited!");
    }
    revng_assert(VisitStack.size() == InStack.size());
    return Inserted;
  };

  void pop() {
    revng_log(Log, "<-- pop(" << VisitStack.back().Node->ID << ')');
    InStack.erase(VisitStack.back().Node);
    VisitStack.pop_back();
    revng_assert(VisitStack.size() == InStack.size());
  };

  bool empty() const { return VisitStack.empty(); }

  bool count(const LTSN *N) const { return InStack.count(N); }

  StackEntry &top() { return VisitStack.back(); }
};

bool RemoveTransitiveInheritanceEdges::runOnTypeSystem(LayoutTypeSystem &TS) {
  if (Log.isEnabled())
    TS.dumpDotOnFile("before-remove-transitive-edges.dot");

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());
  bool Changed = false;

  for (LTSN *Root : llvm::nodes(&TS)) {
    if (not isInheritanceRoot(Root))
      continue;

    revng_log(Log, "Starting DFS from Inheritance Root: " << Root->ID);

    using Edge = LTSN::NeighborsSet::value_type;
    using EdgeInfo = std::tuple<LTSN * /* Src */,
                                LTSN * /* Tgt */,
                                const TypeLinkTag * /* LinkTag */>;
    SmallSet<EdgeInfo, 8> ToErase;

    DFSStack Stack(Root);
    Stack.tryPush(Root);
    while (not Stack.empty()) {
      DFSStack::StackEntry &StackTop = Stack.top();
      LayoutTypeSystemNode *Node = StackTop.Node;
      auto NChildren = StackTop.OrderedChildren.size();
      auto &NextChildPos = StackTop.NextChild;
      revng_log(Log, "# Stack Top: " << Node->ID);

      bool Pushed = false;
      while ((NextChildPos != NChildren) and not Pushed) {
        LTSN *NextChild = StackTop.OrderedChildren[NextChildPos];
        revng_log(Log, "    NextChild: " << NextChild->ID);
        ++NextChildPos;
        Pushed = Stack.tryPush(NextChild);
      }

      if (Pushed)
        continue;

      // Here all the children of Node have been visited.
      revng_log(Log, "# Completed node: " << Node->ID);

      // Loop on all children of the completed node.
      for (LTSN *Child : llvm::children<InheritanceNodeT>(Node)) {
        revng_log(Log, "## Analyzing Inheritance child: " << Child->ID);

        // For each Child, look at the predecessors across inheritance edges
        using InvInheritanceNodeT = llvm::Inverse<InheritanceNodeT>;
        for (const Edge &PredE : children_edges<InvInheritanceNodeT>(Child)) {
          LTSN *Pred = PredE.first;
          revng_log(Log, "### Analyzing Predecessor: " << Pred->ID);
          revng_assert(PredE.second->getKind() == TypeLinkTag::LK_Inheritance);
          if (Pred != Node and Stack.count(Pred)) {
            // This is a predecessor of Node, that is in stack and is not the
            // predecessor from which we arrived here with the visit. Hence, the
            // edge from Pred to Child is a transitive edge, and we must
            // remove it.
            const TypeLinkTag *T = PredE.second;
            revng_log(Log,
                      "#### Found transitive edge: " << Pred->ID << " -> "
                                                     << Child->ID);
            ToErase.insert({ Pred, Child, T });
            revng_assert(Pred->Successors.count(std::make_pair(Child, T)));
            revng_assert(Child->Predecessors.count(std::make_pair(Pred, T)));
          }
        }
      }

      // Ok, we removed all the transitive edges that are incoming into the
      // children of Node and start from any other node above Node in the
      // VisitStack. We can pop this Node.
      Stack.pop();
    }

    if (not ToErase.empty()) {
      Changed = true;

      if (Log.isEnabled()) {
        SmallString<64> Name("edges-removed-node-");
        Name += std::to_string(Root->ID) + ".dot";
        TS.dumpDotOnFile(Name.c_str());
      }

      // Actually remove the edges
      for (auto &[Pred, Child, T] : ToErase) {
        revng_log(Log,
                  "# Removing transitive edge: " << Pred->ID << " -> "
                                                 << Child->ID);
        revng_assert(T->getKind() == TypeLinkTag::LK_Inheritance);
        Edge ChildToPred = std::make_pair(Pred, T);
        bool Erased = Child->Predecessors.erase(ChildToPred);
        revng_assert(Erased);
        Edge PredToChild = std::make_pair(Child, T);
        Erased = Pred->Successors.erase(PredToChild);
        revng_assert(Erased);
      }
    }
  }

  if (Log.isEnabled())
    TS.dumpDotOnFile("after-remove-transitive-edges.dot");

  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyDAG());

  return Changed;
} // namespace dla

} // end namespace dla
