/// \file ASTTree.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>

// Local includes
#include "ASTNode.h"
#include "ASTTree.h"
#include "Utils.h"

// Helper to obtain a unique incremental counter (to give name to sequence
// nodes).
static int Counter = 1;
static std::string getID() {
  return std::to_string(Counter++);
}


SequenceNode *ASTTree::addSequenceNode() {
  ASTNodeList.emplace_back(new SequenceNode("sequence " + getID()));
  return llvm::cast<SequenceNode>(ASTNodeList.back().get());
}

size_t ASTTree::size() { return ASTNodeList.size(); }

void ASTTree::addASTNode(BasicBlockNode *Node,
                         std::unique_ptr<ASTNode>&& ASTObject) {
  NodeASTMap.insert(make_pair(Node, std::move(ASTObject)));
}

ASTNode *ASTTree::findASTNode(BasicBlockNode *BlockNode) {
  ASTNode *ASTPointer = ((NodeASTMap.find(BlockNode))->second).get();
  return ASTPointer;
}

// Helper function that visit an AST tree and creates the sequence nodes
ASTNode *createSequence(ASTTree &Tree, ASTNode *RootNode) {
  SequenceNode *RootSequenceNode = Tree.addSequenceNode();
  RootSequenceNode->addNode(RootNode);

  for (ASTNode *Node : RootSequenceNode->nodes()) {
    if (auto *If = llvm::dyn_cast<IfNode>(Node)) {
      If->setThen(createSequence(Tree, If->getThen()));
      If->setElse(createSequence(Tree, If->getElse()));
    }
    #if 0
  } else if (auto *Code = llvm::dyn_cast<CodeNode>(Node)) {
      // TODO: confirm that doesn't make sense to process a code node.
    } else if (auto *Scs = llvm::dyn_cast<ScsNode>(Node)) {
      // TODO: confirm that this phase is not needed since the processing is
      //       done inside the processing of each SCS region.
    }
    #endif
  }

  return RootSequenceNode;
}

// Helper function which simplifies sequence nodes composed by a single AST
// node.
ASTNode *simplifyAtomicSequence(ASTNode *RootNode) {

  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(RootNode)) {
    if (Sequence->listSize() == 0) {
      RootNode = nullptr;
    } else if (Sequence->listSize() == 1) {
      RootNode = Sequence->getNodeN(0);
      RootNode = simplifyAtomicSequence(RootNode);
    } else {
      for (ASTNode *Node : Sequence->nodes()) {
        Node = simplifyAtomicSequence(Node);
      }
    }
  } else if (auto *If = llvm::dyn_cast<IfNode>(RootNode)) {
    If->setThen(simplifyAtomicSequence(If->getThen()));
    If->setElse(simplifyAtomicSequence(If->getElse()));
  }
  #if 0
} else if (auto *Scs = llvm::dyn_cast<ScsNode>(RootNode)) {
    // TODO: check if this is not needed as the simplification is done for each
    //       SCS region.
  }
  #endif

  return RootNode;
}
