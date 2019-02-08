/// \file Flattening.cpp
/// \brief Helper functions for flattening the RegionCFGTree after combing

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// revng libraries includes
#include "revng/Support/Debug.h"

// local libraries includes
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"
#include "revng-c/RestructureCFGPass/Utils.h"

// local includes
#include "Flattening.h"

Logger<> FlattenLog("flattening");

void flattenRegionCFGTree(RegionCFG &Root) {

  std::set<BasicBlockNode *> CollapsedNodes;
  std::set<BasicBlockNode *> NodesToRemove;

  for (BasicBlockNode *CollapsedNode : Root)
    if (CollapsedNode->isCollapsed())
      CollapsedNodes.insert(CollapsedNode);

  while (CollapsedNodes.size()) {

    for (BasicBlockNode *CollapsedNode : CollapsedNodes) {
      revng_assert(CollapsedNode->successor_size() <= 1);
      RegionCFG *CollapsedRegion = CollapsedNode->getCollapsedCFG();
      BasicBlockNode *OldEntry = &CollapsedRegion->getEntryNode();
      // move nodes to root RegionCFG
      RegionCFG::BBNodeMap SubstitutionMap{};
      using IterT = RegionCFG::links_container::iterator;
      using MovedIterRange = llvm::iterator_range<IterT>;
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

      // Fix predecessors
      BasicBlockNode *Entry = SubstitutionMap.at(OldEntry);
      for (BasicBlockNode *Pred : CollapsedNode->predecessors())
        moveEdgeTarget({Pred, CollapsedNode}, Entry);

      // Fix successors and loops
      BasicBlockNode *Succ = *CollapsedNode->successors().begin();
      for (std::unique_ptr<BasicBlockNode> &UniqueBBNode : MovedRange) {
        if (UniqueBBNode->isBreak() or UniqueBBNode->isContinue()) {
          BasicBlockNode *NewTarget = UniqueBBNode->isBreak() ?  Succ : Entry;
          for (BasicBlockNode *Pred : UniqueBBNode->predecessors())
            moveEdgeTarget({Pred, UniqueBBNode.get()}, NewTarget);
          // breaks and continues need to be removed
          NodesToRemove.insert(UniqueBBNode.get());
        }
      }
      // also collapsed nodes need to be removed
      NodesToRemove.insert(CollapsedNode);
    }

    // remove superfluous nodes
    for (BasicBlockNode *BBNode: NodesToRemove)
      Root.removeNode(BBNode);
    NodesToRemove.clear();

    // check if there are remaining collapsed nodes
    CollapsedNodes.clear();
    for (BasicBlockNode *CollapsedNode : Root)
      if (CollapsedNode->isCollapsed())
        CollapsedNodes.insert(CollapsedNode);

  }
}
