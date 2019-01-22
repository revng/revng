/// \file ASTNode.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <fstream>
#include <iostream>

// Local libraries includes
#include "revng-c/RestructureCFGPass/ASTNode.h"

using namespace llvm;

void IfNode::addConditionalNodesFrom(IfNode *Other) {
  for (BasicBlockNode *Node : Other->conditionalNodes()) {
    ConditionalNodes.push_back(Node);
  }
}

bool CodeNode::isEqual(ASTNode *Node) {
  if (auto *OtherCode = dyn_cast<CodeNode>(Node)) {
    if ((getOriginalBB() != nullptr)
        and (getOriginalBB() == OtherCode->getOriginalBB())) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

bool IfNode::isEqual(ASTNode *Node) {
  if (auto *OtherIf = dyn_cast<IfNode>(Node)) {
    if ((getOriginalBB() != nullptr)
        and (getOriginalBB() == OtherIf->getOriginalBB())) {

      // TODO: this is necessary since we may not have one between `then` or
      //       `else` branches, refactor in a more elegant way
      bool ComparisonState = true;
      if (hasThen()) {
        ComparisonState = Then->isEqual(OtherIf->getThen());
      }
      if (hasElse()) {
        ComparisonState = Else->isEqual(OtherIf->getElse());
      }
      return ComparisonState;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

bool ScsNode::isEqual(ASTNode *Node) {
  if (auto *OtherScs = dyn_cast<ScsNode>(Node)) {
    if (Body->isEqual(OtherScs->getBody())) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

bool SequenceNode::isEqual(ASTNode *Node) {
  if (auto *OtherSequence = dyn_cast<SequenceNode>(Node)) {
    bool ComparisonState = true;
    int FirstDimension = NodeList.size();
    int SecondDimension = OtherSequence->listSize();
    if (FirstDimension != SecondDimension) {
      ComparisonState = false;
    }

    // Continue the comparison only if the sequence node size are the same
    if (ComparisonState) {
      assert (FirstDimension == SecondDimension);
      for (int I = 0; I < FirstDimension; I++) {
        ASTNode *FirstNode = getNodeN(I);
        ASTNode *SecondNode = OtherSequence->getNodeN(I);

        // As soon as two nodes does not match, exit and make the comparison
        // fail
        if (!FirstNode->isEqual(SecondNode)) {
          ComparisonState = false;
          break;
        }
      }
    }

    return ComparisonState;
  } else {
    return false;
  }
}

void CodeNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void IfNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";

  // For the label of the If node go take all the nodes in the list
  std::string ConditionalNames;
  for (BasicBlockNode *Conditional : this->conditionalNodes()) {
    ConditionalNames += Conditional->getNameStr() + ", ";
  }
  ConditionalNames.pop_back();
  ConditionalNames.pop_back();

  ASTFile << "label=\"" << ConditionalNames;
  ASTFile << "\"";
  ASTFile << ",shape=\"invhouse\",color=\"blue\"];\n";

  if (this->getThen() != nullptr) {
    ASTFile << "\"" << this->getName() << "\""
        << " -> \"" << this->getThen()->getName() << "\""
        << " [color=green,label=\"then\"];\n";
    this->getThen()->dump(ASTFile);
  }

  if (this->getElse() != nullptr) {
    ASTFile << "\"" << this->getName() << "\""
        << " -> \"" << this->getElse()->getName() << "\""
        << " [color=green,label=\"else\"];\n";
    this->getElse()->dump(ASTFile);
  }
}

void ScsNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"circle\",color=\"black\"];\n";

  revng_assert(this->getBody() != nullptr);
  ASTFile << "\"" << this->getName() << "\""
      << " -> \"" << this->getBody()->getName() << "\""
      << " [color=green,label=\"body\"];\n";
  this->getBody()->dump(ASTFile);
}

void SequenceNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"box\",color=\"black\"];\n";

  for (ASTNode *Successor : this->nodes()) {
    ASTFile << "\"" << this->getName() << "\""
        << " -> \"" << Successor->getName() << "\""
        << " [color=green,label=\"elem\"];\n";
    Successor->dump(ASTFile);
  }
}
