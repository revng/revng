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

void IfNode::updateCondExprPtr(ExprNodeMap &Map) {
  revng_assert(ConditionExpression != nullptr);
  ConditionExpression = Map[ConditionExpression];
}

void ContinueNode::addComputationIfNode(IfNode *ComputationIfNode) {
  revng_assert(ComputationIf == nullptr);
  ComputationIf = ComputationIfNode;
}

IfNode *ContinueNode::getComputationIfNode() const {
  revng_assert(ComputationIf != nullptr);
  return ComputationIf;
}

// #### updateASTNodesPointers methods ####

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

void SequenceNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  // Update all the pointers of the sequence node.
  for (auto NodeIt = NodeList.begin(); NodeIt != NodeList.end(); NodeIt++) {
    ASTNode *Node = *NodeIt;
    revng_assert(SubstitutionMap.count(Node) != 0);
    ASTNode *NewNode = SubstitutionMap[Node];
    *NodeIt = NewNode;
  }
}

void SwitchNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  for (auto &Case : CaseVec)
    Case = SubstitutionMap.at(Case);
}

// #### isEqual methods ####

template<typename SwitchNodeType>
typename SwitchNodeType::case_value
getCaseValueN(const SwitchNodeType *S,
              typename SwitchNodeType::case_container::size_type N) {
  return cast<SwitchNodeType>(S)->getCaseValueN(N);
}

bool SwitchNode::hasEqualCaseValues(const SwitchNode *Node) const {
  if (this->getKind() != Node->getKind())
    return false;
  revng_assert(CaseSize() == Node->CaseSize());

  auto *SwitchDispatcherThis = dyn_cast<SwitchDispatcherNode>(this);
  auto *RegSwitchThis = dyn_cast<RegularSwitchNode>(this);
  auto *SwitchDispatcherOther = dyn_cast<SwitchDispatcherNode>(Node);
  auto *RegSwitchOther = dyn_cast<RegularSwitchNode>(Node);
  for (case_container::size_type I = 0; I < CaseSize(); I++) {
    if (SwitchDispatcherThis != nullptr) {
      uint64_t ThisCase = getCaseValueN(SwitchDispatcherThis, I);
      uint64_t OtherCase = getCaseValueN(SwitchDispatcherOther, I);
      if (ThisCase != OtherCase)
        return false;
    } else {
      revng_assert(RegSwitchThis != nullptr);
      SmallPtrSet<ConstantInt *, 1> ThisCaseSet = getCaseValueN(RegSwitchThis,
                                                                I);
      SmallPtrSet<ConstantInt *, 1> OtherCaseSet = getCaseValueN(RegSwitchOther,
                                                                 I);
      if (ThisCaseSet != OtherCaseSet)
        return false;
    }
  }

  return true;
}

bool SwitchNode::isEqual(const ASTNode *Node) const {
  auto *OtherSwitch = dyn_cast_or_null<RegularSwitchNode>(Node);
  if (OtherSwitch == nullptr)
    return false;

  ASTNode *OtherDefault = OtherSwitch->getDefault();
  ASTNode *ThisDefault = this->getDefault();
  if ((OtherDefault == nullptr) != (ThisDefault == nullptr))
    return false;
  if (ThisDefault and not ThisDefault->isEqual(OtherDefault))
    return false;

  case_container::size_type FirstDimension = CaseSize();
  case_container::size_type SecondDimension = OtherSwitch->CaseSize();
  // Continue the comparison only if the sequence node size are the same
  if (FirstDimension != SecondDimension)
    return false;

  // TODO: here we assume that the cases are in the same order, which has been
  // true until now but it's not guaranteed. We should enforce that this
  // equality check is performed on cases with the same values.
  if (not hasEqualCaseValues(OtherSwitch))
    return false;

  for (case_container::size_type I = 0; I < FirstDimension; I++) {
    ASTNode *FirstNode = getCaseN(I);
    ASTNode *SecondNode = OtherSwitch->getCaseN(I);

    // As soon as two nodes does not match, exit and make the comparison fail
    if (!FirstNode->isEqual(SecondNode))
      return false;
  }

  return true;
}

bool CodeNode::isEqual(const ASTNode *Node) const {
  auto *OtherCode = dyn_cast_or_null<CodeNode>(Node);
  if (OtherCode == nullptr)
    return false;

  return (getOriginalBB() != nullptr)
         and (getOriginalBB() == OtherCode->getOriginalBB());
}

bool IfNode::isEqual(const ASTNode *Node) const {
  auto *OtherIf = dyn_cast_or_null<IfNode>(Node);
  if (OtherIf == nullptr)
    return false;

  if ((getOriginalBB() != nullptr)
      and (getOriginalBB() == OtherIf->getOriginalBB())) {

    // TODO: this is necessary since we may not have one between `then` or
    //       `else` branches, refactor in a more elegant way
    bool ComparisonState = true;
    if (hasThen())
      ComparisonState = Then->isEqual(OtherIf->getThen());
    if (hasElse())
      ComparisonState = Else->isEqual(OtherIf->getElse());
    return ComparisonState;
  }
  return false;
}

bool ScsNode::isEqual(const ASTNode *Node) const {
  auto *OtherScs = dyn_cast_or_null<ScsNode>(Node);
  if (OtherScs == nullptr)
    return false;

  return Body->isEqual(OtherScs->getBody());
}

bool SetNode::isEqual(const ASTNode *Node) const {
  auto *OtherSet = dyn_cast_or_null<SetNode>(Node);
  if (OtherSet == nullptr)
    return false;
  return StateVariableValue == OtherSet->getStateVariableValue();
}

bool SequenceNode::isEqual(const ASTNode *Node) const {
  auto *OtherSequence = dyn_cast_or_null<SequenceNode>(Node);
  if (OtherSequence == nullptr)
    return false;

  links_container::size_type FirstDimension = NodeList.size();
  links_container::size_type SecondDimension = OtherSequence->listSize();
  // Continue the comparison only if the sequence node size are the same
  if (FirstDimension != SecondDimension)
    return false;

  revng_assert(FirstDimension == SecondDimension);
  for (links_container::size_type I = 0; I < FirstDimension; I++) {
    ASTNode *FirstNode = getNodeN(I);
    ASTNode *SecondNode = OtherSequence->getNodeN(I);

    // As soon as two nodes does not match, exit and make the comparison fail
    if (!FirstNode->isEqual(SecondNode))
      return false;
  }
  return true;
}

// #### Dump methods ####

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
  // revng_assert(this->getBody() != nullptr);
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

void RegularSwitchNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"hexagon\",color=\"black\"];\n";

  case_container::size_type CaseIndex = 0;
  for (ASTNode *Case : this->unordered_cases()) {
    ASTFile << "\"" << this->getName() << "\""
            << " -> \"" << Case->getName() << "\""
            << " [color=green,label=\"case ";

    // Cases can now be sets of cases, we need to print all of them on a edge.
    for (auto *CaseConstantInt : CaseValueVec[CaseIndex]) {
      uint64_t CaseVal = CaseConstantInt->getZExtValue();
      ASTFile << CaseVal << ",";
    }

    // Close the line.
    ASTFile << "\"];\n";

    // Continue dumping the children of the switch node.
    Case->dump(ASTFile);
    ++CaseIndex;
  }
  if (ASTNode *Default = this->getDefault()) {
    ASTFile << "\"" << this->getName() << "\""
            << " -> \"" << Default->getName() << "\""
            << " [color=green,label=\"default\"];\n";
    Default->dump(ASTFile);
  }
}

void BreakNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"loop break\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void SwitchBreakNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"switch break\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void ContinueNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"continue\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void SetNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void SwitchDispatcherNode::dump(std::ofstream &ASTFile) {
  ASTFile << "\"" << this->getName() << "\" [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"hexagon\",color=\"black\"];\n";

  case_container::size_type CaseIndex = 0;
  for (ASTNode *Case : this->unordered_cases()) {
    uint64_t CaseVal = CaseValueVec[CaseIndex];
    ASTFile << "\"" << this->getName() << "\""
            << " -> \"" << Case->getName() << "\""
            << " [color=green,label=\"case " << CaseVal << "\"];\n";
    Case->dump(ASTFile);
    ++CaseIndex;
  }
  if (ASTNode *Default = this->getDefault()) {
    ASTFile << "\"" << this->getName() << "\""
            << " -> \"" << Default->getName() << "\""
            << " [color=green,label=\"default\"];\n";
    Default->dump(ASTFile);
  }
}

void ASTNode::deleteASTNode(ASTNode *A) {
  switch (A->getKind()) {
  case NodeKind::NK_Code:
    delete static_cast<CodeNode *>(A);
    break;
  case NodeKind::NK_Break:
    delete static_cast<BreakNode *>(A);
    break;
  case NodeKind::NK_Continue:
    delete static_cast<ContinueNode *>(A);
    break;
  // ---- IfNode kinds
  case NodeKind::NK_If:
    delete static_cast<IfNode *>(A);
    break;
  // ---- end IfNode kinds
  case NodeKind::NK_Scs:
    delete static_cast<ScsNode *>(A);
    break;
  case NodeKind::NK_List:
    delete static_cast<SequenceNode *>(A);
    break;
  // ---- SwitchNode kinds
  case NodeKind::NK_SwitchRegular:
    delete static_cast<RegularSwitchNode *>(A);
    break;
  case NodeKind::NK_SwitchDispatcher:
    delete static_cast<SwitchDispatcherNode *>(A);
    break;
  // ---- end SwitchNode kinds
  case NodeKind::NK_SwitchBreak:
    delete static_cast<SwitchBreakNode *>(A);
    break;
  case NodeKind::NK_Set:
    delete static_cast<SetNode *>(A);
    break;
  }
}

void SwitchNode::removeCaseN(case_container::size_type N) {
  revng_assert(N < CaseSize());
  CaseVec.erase(CaseVec.begin() + N);

  // Remove also the counterpart of the N-th case node even in the
  // `CaseValueVec` field of the subclass.
  switch (this->getKind()) {
  case NodeKind::NK_SwitchRegular: {
    RegularSwitchNode *SwitchR = static_cast<RegularSwitchNode *>(this);
    auto &CaseValueVec = SwitchR->CaseValueVec;
    CaseValueVec.erase(CaseValueVec.begin() + N);
  } break;
  case NodeKind::NK_SwitchDispatcher: {
    SwitchDispatcherNode *SwitchD = static_cast<SwitchDispatcherNode *>(this);
    auto &CaseValueVec = SwitchD->CaseValueVec;
    CaseValueVec.erase(CaseValueVec.begin() + N);
  } break;
  default:
    revng_unreachable();
  }
}
