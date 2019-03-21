/// \file ASTNode.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>
#include <fstream>
#include <iostream>

// LLVM includes
#include <llvm/IR/Constants.h>

// Local libraries includes
#include "revng-c/RestructureCFGPass/ASTNode.h"

using namespace llvm;

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

void CodeNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  // A CodeNode should not contain any reference to other AST nodes, we do not
  // need to apply updates.
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

void IfNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  // Update the pointers to the `then` and `else` branches.
  if (hasThen()) {
    revng_assert(SubstitutionMap.count(Then) != 0);
    Then = SubstitutionMap[Then];
  }

  if (hasElse()) {
    revng_assert(SubstitutionMap.count(Else) != 0);
    Else = SubstitutionMap[Else];
  }
}

void IfNode::updateCondExprPtr(ExprNodeMap &Map) {
  if (not llvm::isa<IfCheckNode>(this)) {
      revng_assert(ConditionExpression != nullptr);
      ConditionExpression = Map[ConditionExpression];
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

void ScsNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  // The link to the body AST node should be fixed explicitly during the
  // flattening.
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
      revng_assert(FirstDimension == SecondDimension);
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

void SequenceNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  // Update all the pointers of the sequence node.
  for (auto NodeIt = NodeList.begin(); NodeIt != NodeList.end(); NodeIt++) {
    ASTNode *Node = *NodeIt;
    revng_assert(SubstitutionMap.count(Node) != 0);
    ASTNode *NewNode = SubstitutionMap[Node];
    *NodeIt = NewNode;
  }
}

void BreakNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"break\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void ContinueNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"continue\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void ContinueNode::addComputationIfNode(IfNode *ComputationIfNode) {
  revng_assert(ComputationIf == nullptr);
  ComputationIf = ComputationIfNode;
}

IfNode *ContinueNode::getComputationIfNode() {
  revng_assert(ComputationIf != nullptr);
  return ComputationIf;
}

void CodeNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void IfNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";

  // TODO: Implement the printing of the conditional expression for the if.

  // ASTFile << "label=\"" << ConditionalNames;
  ASTFile << "label=\"" << this->getName();
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

  // After do-while and while match loop nodes could be empty
  //revng_assert(this->getBody() != nullptr);
  if (this->getBody() != nullptr) {
    ASTFile << "\"" << this->getName() << "\""
            << " -> \"" << this->getBody()->getName() << "\""
            << " [color=green,label=\"body\"];\n";
    this->getBody()->dump(ASTFile);
  }
}

void SequenceNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"box\",color=\"black\"];\n";

  int SuccessorIndex = 0;
  for (ASTNode *Successor : this->nodes()) {
    ASTFile << "\"" << this->getName() << "\""
            << " -> \"" << Successor->getName() << "\""
            << " [color=green,label=\"elem " << SuccessorIndex << "\"];\n";
    Successor->dump(ASTFile);
    SuccessorIndex += 1;
  }
}

void SwitchNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"hexagon\",color=\"black\"];\n";

  int CaseIndex = 0;
  for (auto Case : this->cases()) {
    uint64_t CaseVal = Case.first->getZExtValue();
    ASTFile << "\"" << this->getName() << "\""
            << " -> \"" << Case.second->getName() << "\""
            << " [color=green,label=\"case " << CaseVal << "\"];\n";
    Case.second->dump(ASTFile);
    CaseIndex += 1;
  }
}

bool SwitchNode::isEqual(ASTNode *Node) {
  if (auto *OtherSwitch = dyn_cast<SwitchNode>(Node)) {
    bool ComparisonState = true;
    int FirstDimension = CaseList.size();
    int SecondDimension = OtherSwitch->CaseSize();
    if (FirstDimension != SecondDimension) {
      ComparisonState = false;
    }

    // Continue the comparison only if the sequence node size are the same
    if (ComparisonState) {
      revng_assert(FirstDimension == SecondDimension);
      for (int I = 0; I < FirstDimension; I++) {
        ASTNode *FirstNode = getCaseN(I);
        ASTNode *SecondNode = OtherSwitch->getCaseN(I);

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

void SwitchNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  // Update all the case pointers.
  for (auto &Case : CaseList) {
    Case.second = SubstitutionMap.at(Case.second);
  }
}

bool SetNode::isEqual(ASTNode *Node) {
  if (auto *OtherSet = dyn_cast<SetNode>(Node)) {
    if (StateVariableValue == OtherSet->getStateVariableValue()) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void SetNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void SetNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  // A SetNode should not contain any reference to other AST nodes.
}

void IfCheckNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"invtrapezium\",color=\"red\"];\n";

  // We may not have one between the `then` and `else` branches, since its
  // function could be replaced by the fallthrough AST node.
  if (this->getThen() != nullptr) {
    ASTFile << "\"" << this->getName() << "\""
            << " -> \"" << this->getThen()->getName() << "\""
            << " [color=green,label=\"then\"];\n";
    this->getThen()->dump(ASTFile);
  }

  if (this->getElse() != nullptr) {
    ASTFile << "\"" << this->getName() << "\""
            << " -> \"" << this->getElse()->getName() << "\""
            << " [color=red,label=\"else\"];\n";
    this->getElse()->dump(ASTFile);
  }
}

void SwitchCheckNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"hexagon\",color=\"black\"];\n";

  int CaseIndex = 0;
  for (auto Case : this->cases()) {
    uint64_t CaseVal = Case.first;
    ASTFile << "\"" << this->getName() << "\""
            << " -> \"" << Case.second->getName() << "\""
            << " [color=green,label=\"case " << CaseVal << "\"];\n";
    Case.second->dump(ASTFile);
    CaseIndex += 1;
  }
}

bool SwitchCheckNode::isEqual(ASTNode *Node) {
  if (auto *OtherSwitch = dyn_cast<SwitchCheckNode>(Node)) {
    bool ComparisonState = true;
    int FirstDimension = CaseList.size();
    int SecondDimension = OtherSwitch->CaseSize();
    if (FirstDimension != SecondDimension) {
      ComparisonState = false;
    }

    // Continue the comparison only if the sequence node size are the same
    if (ComparisonState) {
      revng_assert(FirstDimension == SecondDimension);
      for (int I = 0; I < FirstDimension; I++) {
        ASTNode *FirstNode = getCaseN(I);
        ASTNode *SecondNode = OtherSwitch->getCaseN(I);

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

void SwitchCheckNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  // Update all the case pointers.
  for (auto &Case : CaseList) {
    Case.second = SubstitutionMap.at(Case.second);
  }
}
