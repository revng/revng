#ifndef REVNGC_RESTRUCTURE_CFG_FLATTENINGIMPL_H
#define REVNGC_RESTRUCTURE_CFG_FLATTENINGIMPL_H
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/ADT/SmallVector.h>

// revng libraries includes
#include <revng/Support/Debug.h>

// Local libraries includes
#include "revng-c/RestructureCFGPass/Flattening.h"
#include "revng-c/RestructureCFGPass/RegionCFGTreeBB.h"
#include "revng-c/RestructureCFGPass/Utils.h"

template<class NodeT>
inline void flattenRegionCFGTree(RegionCFG<NodeT> &Root) {

  using BasicBlockNodeT = typename BasicBlockNode<NodeT>::BasicBlockNodeT;
  using EdgeDescriptor = typename BasicBlockNode<NodeT>::EdgeDescriptor;

  std::set<BasicBlockNodeT *> CollapsedNodes;
  std::set<BasicBlockNodeT *> NodesToRemove;

  for (BasicBlockNodeT *CollapsedNode : Root)
    if (CollapsedNode->isCollapsed())
      CollapsedNodes.insert(CollapsedNode);

  while (CollapsedNodes.size()) {

    for (BasicBlockNodeT *CollapsedNode : CollapsedNodes) {
      revng_assert(CollapsedNode->successor_size() <= 1);
      RegionCFG<NodeT> *CollapsedRegion = CollapsedNode->getCollapsedCFG();
      BasicBlockNodeT *OldEntry = &CollapsedRegion->getEntryNode();
      // move nodes to root RegionCFG
      typename RegionCFG<NodeT>::BBNodeMap SubstitutionMap{};
      using IterT = typename RegionCFG<NodeT>::links_container::iterator;
      using MovedIterRange = typename llvm::iterator_range<IterT>;
      MovedIterRange MovedRange = Root.copyNodesAndEdgesFrom(CollapsedRegion,
                                                             SubstitutionMap);

      // Obtain a reference to the root AST node.
      ASTTree &RootAST = Root.getAST();
      ASTNode *ASTEntry = RootAST.copyASTNodesFrom(CollapsedRegion->getAST(),
                                                   SubstitutionMap);

      // Fix body field of predecessor AST nodes
      ASTNode *SCSNode = RootAST.findASTNode(CollapsedNode);
      revng_assert((SCSNode != nullptr) and (llvm::isa<ScsNode>(SCSNode)));
      ScsNode *SCS = llvm::cast<ScsNode>(SCSNode);
      SCS->setBody(ASTEntry);

      llvm::SmallVector<EdgeDescriptor, 4> ToMove;

      // Fix predecessors
      BasicBlockNodeT *Entry = SubstitutionMap.at(OldEntry);
      for (BasicBlockNodeT *Pred : CollapsedNode->predecessors())
        ToMove.push_back({ Pred, CollapsedNode });
      for (EdgeDescriptor &Edge : ToMove)
        moveEdgeTarget(Edge, Entry);

      // Fix successors and loops
      BasicBlockNodeT *Succ = *CollapsedNode->successors().begin();
      for (std::unique_ptr<BasicBlockNodeT> &UniqueBBNode : MovedRange) {
        if (UniqueBBNode->isBreak() or UniqueBBNode->isContinue()) {
          BasicBlockNodeT *NewTarget = UniqueBBNode->isBreak() ? Succ : Entry;
          ToMove.clear();
          for (BasicBlockNodeT *Pred : UniqueBBNode->predecessors())
            ToMove.push_back({ Pred, UniqueBBNode.get() });
          for (EdgeDescriptor &Edge : ToMove)
            moveEdgeTarget(Edge, NewTarget);
          // breaks and continues need to be removed
          NodesToRemove.insert(UniqueBBNode.get());
        }
      }
      // also collapsed nodes need to be removed
      NodesToRemove.insert(CollapsedNode);
    }

    // remove superfluous nodes
    for (BasicBlockNodeT *BBNode : NodesToRemove)
      Root.removeNode(BBNode);
    NodesToRemove.clear();

    // check if there are remaining collapsed nodes
    CollapsedNodes.clear();
    for (BasicBlockNodeT *CollapsedNode : Root)
      if (CollapsedNode->isCollapsed())
        CollapsedNodes.insert(CollapsedNode);
  }
}

#endif // REVNGC_RESTRUCTURE_CFG_FLATTENINGIMPL_H
