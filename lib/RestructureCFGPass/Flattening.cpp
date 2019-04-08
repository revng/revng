/// \file Flattening.cpp
/// \brief Helper functions for flattening the RegionCFGTree after combing

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/ADT/SmallVector.h>

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

      using EdgeDescriptor = std::pair<BasicBlockNode *, BasicBlockNode *>;
      llvm::SmallVector<EdgeDescriptor, 4> ToMove;

      // Fix predecessors
      BasicBlockNode *Entry = SubstitutionMap.at(OldEntry);
      for (BasicBlockNode *Pred : CollapsedNode->predecessors())
        ToMove.push_back({ Pred, CollapsedNode });
      for (EdgeDescriptor &Edge : ToMove)
        moveEdgeTarget(Edge, Entry);

      // Fix successors and loops
      BasicBlockNode *Succ = *CollapsedNode->successors().begin();
      for (std::unique_ptr<BasicBlockNode> &UniqueBBNode : MovedRange) {
        if (UniqueBBNode->isBreak() or UniqueBBNode->isContinue()) {
          BasicBlockNode *NewTarget = UniqueBBNode->isBreak() ? Succ : Entry;
          ToMove.clear();
          for (BasicBlockNode *Pred : UniqueBBNode->predecessors())
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
    for (BasicBlockNode *BBNode : NodesToRemove)
      Root.removeNode(BBNode);
    NodesToRemove.clear();

    // check if there are remaining collapsed nodes
    CollapsedNodes.clear();
    for (BasicBlockNode *CollapsedNode : Root)
      if (CollapsedNode->isCollapsed())
        CollapsedNodes.insert(CollapsedNode);
  }

  NodesToRemove.clear();
  std::vector<BasicBlockNode *> SetNodes;
  // After we've finished the flattening, remove all the Set nodes and all the
  // chains of Switch nodes. This is beneficial, because Set and Check nodes
  // added by the combing do actually introduce new control flow that was not
  // present in the original LLVM IR. We want to avoid this because adding
  // non-existing control flow may hamper the results of future analyses
  // performed on the LLVM IR after the combing.
  for (BasicBlockNode *Node : Root) {
    switch (Node->getNodeType()) {
    case BasicBlockNode::Type::Set: {
      SetNodes.push_back(Node);
    } break;
    case BasicBlockNode::Type::Check: {
      revng_assert(Node->predecessor_size() != 0);
      BasicBlockNode *Pred = Node->getPredecessorI(0);
      if (Pred->isCheck()) {
        revng_assert(Node->predecessor_size() == 1);
        continue;
      }

      // Here Node is the head of a chain of Check nodes, and all its
      // predecessors are Set nodes, or they are dummy nodes, the predecessor
      // of which must in turn be Check nodes or dummy nodes.
      std::multimap<unsigned, BasicBlockNode *> VarToSet;
      std::vector<BasicBlockNode *> Candidates;

      // Iterative exploration going upwards from the check node searching for
      // the set nodes.
      for (BasicBlockNode *Pred : Node->predecessors()) {
        Candidates.push_back(Pred);
      }
      while (Candidates.size() > 0) {
        BasicBlockNode *Candidate = Candidates.back();
        Candidates.pop_back();
        if (Candidate->isSet()) {

          // If the predecessor is a set node add it for later processing.
          unsigned SetID = Candidate->getStateVariableValue();
          VarToSet.insert({ SetID, Candidate });
        } else if (Candidate->isEmpty()) {

          // If the predecessor is a dummy node, enqueue all its predecessor
          // for processing, after verifying that they are in turn either set
          // or dummy nodes.
          std::vector<EdgeDescriptor> EdgesToMove;
          for (BasicBlockNode *Pred : Candidate->predecessors()) {
            revng_assert(Pred->isSet() or Pred->isEmpty());
            Candidates.push_back(Pred);
            EdgesToMove.push_back({ Pred, Candidate });
          }

          // Remove the dummy node when we have finished.
          NodesToRemove.insert(Candidate);

          // Remove the edges between the set nodes to the dummy node.
          for (EdgeDescriptor &Edge : EdgesToMove) {
            moveEdgeTarget(Edge, Node);
          }
        } else {
          revng_abort("Wrong ascending path towards set nodes");
        }
      }

      BasicBlockNode *Check = Node;
      BasicBlockNode *False = nullptr;
      while (1) {
        NodesToRemove.insert(Check);
        unsigned CheckId = Check->getStateVariableValue();
        BasicBlockNode *True = Check->getTrue();
        auto Range = VarToSet.equal_range(CheckId);
        for (auto &Pair : llvm::make_range(Range.first, Range.second)) {
          BasicBlockNode *SetNode = Pair.second;
          moveEdgeTarget({ SetNode, Node }, True);
        }
        False = Check->getFalse();
        if (not False->isCheck())
          break;
        Check = False;
      }
      auto Range = VarToSet.equal_range(0);
      for (auto &Pair : llvm::make_range(Range.first, Range.second)) {
        BasicBlockNode *SetNode = Pair.second;
        moveEdgeTarget({ SetNode, Node }, False);
      }
    } break;
    }
  }

  // Connect all the predecessors of the set nodes directly to the original
  // successor, ignoring the set and check nodes.
  for (BasicBlockNode *SetNode : SetNodes) {
    revng_assert(SetNode->successor_size() == 1);
    BasicBlockNode *Succ = SetNode->getSuccessorI(0);
    for (BasicBlockNode *Pred : SetNode->predecessors())
      moveEdgeTarget({ Pred, SetNode }, Succ);
    NodesToRemove.insert(SetNode);
  }
  for (BasicBlockNode *BBNode : NodesToRemove)
    Root.removeNode(BBNode);
}
