/// \file ASTNode.cpp

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <cstdlib>
#include <fstream>
#include <iostream>

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"

#include "revng-c/RestructureCFG/ASTNode.h"

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

void ScsNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  if (RelatedCondition)
    Body = SubstitutionMap.at(RelatedCondition);
  revng_assert(Body);
  Body = SubstitutionMap.at(Body);
}

void SequenceNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  // Update all the pointers of the sequence node.
  for (auto NodeIt = NodeVec.begin(); NodeIt != NodeVec.end(); NodeIt++) {
    ASTNode *Node = *NodeIt;
    revng_assert(SubstitutionMap.count(Node) != 0);
    ASTNode *NewNode = SubstitutionMap[Node];
    *NodeIt = NewNode;
  }
}

void SwitchNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  for (auto &LabelCasePair : LabelCaseVec)
    LabelCasePair.second = SubstitutionMap.at(LabelCasePair.second);

  if (Default != nullptr)
    Default = SubstitutionMap.at(Default);
}

// #### isEqual methods ####

template<typename SwitchNodeType>
typename SwitchNodeType::case_value
getCaseValueN(const SwitchNodeType *S,
              typename SwitchNodeType::case_container::size_type N) {
  return cast<SwitchNodeType>(S)->getCaseValueN(N);
}

bool SwitchNode::nodeIsEqual(const ASTNode *Node) const {
  auto *OtherSwitch = dyn_cast_or_null<SwitchNode>(Node);
  if (OtherSwitch == nullptr)
    return false;

  if (getOriginalBB() != Node->getOriginalBB())
    return false;

  ASTNode *OtherDefault = OtherSwitch->getDefault();
  ASTNode *ThisDefault = this->getDefault();
  if ((OtherDefault == nullptr) != (ThisDefault == nullptr))
    return false;

  if (ThisDefault and not ThisDefault->isEqual(OtherDefault))
    return false;

  // Continue the comparison only if the sequence node size are the same
  if (LabelCaseVec.size() != OtherSwitch->LabelCaseVec.size())
    return false;

  for (const auto &PairOfPairs :
       llvm::zip_first(cases_const_range(), OtherSwitch->cases_const_range())) {
    const auto &[ThisCase, OtherCase] = PairOfPairs;
    const auto &[ThisCaseLabel, ThisCaseChild] = ThisCase;
    const auto &[OtherCaseLabel, OtherCaseChild] = OtherCase;

    if (ThisCaseLabel != OtherCaseLabel)
      return false;

    if (not ThisCaseChild->isEqual(OtherCaseChild))
      return false;
  }

  return true;
}

bool CodeNode::nodeIsEqual(const ASTNode *Node) const {
  auto *OtherCode = dyn_cast_or_null<CodeNode>(Node);
  if (OtherCode == nullptr)
    return false;

  return (getOriginalBB() != nullptr)
         and (getOriginalBB() == OtherCode->getOriginalBB());
}

bool IfNode::nodeIsEqual(const ASTNode *Node) const {
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

bool ScsNode::nodeIsEqual(const ASTNode *Node) const {
  auto *OtherScs = dyn_cast_or_null<ScsNode>(Node);
  if (OtherScs == nullptr)
    return false;

  return Body->isEqual(OtherScs->getBody());
}

bool SetNode::nodeIsEqual(const ASTNode *Node) const {
  auto *OtherSet = dyn_cast_or_null<SetNode>(Node);
  if (OtherSet == nullptr)
    return false;
  return StateVariableValue == OtherSet->getStateVariableValue();
}

bool SequenceNode::nodeIsEqual(const ASTNode *Node) const {
  auto *OtherSequence = dyn_cast_or_null<SequenceNode>(Node);
  if (OtherSequence == nullptr)
    return false;

  links_container::size_type FirstDimension = NodeVec.size();
  links_container::size_type SecondDimension = OtherSequence->length();
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

void CodeNode::dump(llvm::raw_fd_ostream &ASTFile) {
  ASTFile << "node_" << this->getID() << " [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void CodeNode::dumpEdge(llvm::raw_fd_ostream &ASTFile) {
  // Do nothing, we don't have outgoing edges.
}

static std::string printBBName(ExprNode *Condition) {
  if (auto *Atomic = llvm::dyn_cast<AtomicNode>(Condition))
    return Atomic->getConditionalBasicBlock()->getName().str();

  if (auto *And = llvm::dyn_cast<AndNode>(Condition)) {
    return "(" + printBBName(And->getInternalNodes().first) + ") and ("
           + printBBName(And->getInternalNodes().second) + ")";
  }

  if (auto *Or = llvm::dyn_cast<OrNode>(Condition)) {
    return "(" + printBBName(Or->getInternalNodes().first) + ") or  ("
           + printBBName(Or->getInternalNodes().second) + ")";
  }

  if (auto *Not = llvm::dyn_cast<NotNode>(Condition)) {
    return "not (" + printBBName(Not->getNegatedNode()) + ")";
  }

  revng_abort();
}

void IfNode::dump(llvm::raw_fd_ostream &ASTFile) {
  ASTFile << "node_" << this->getID() << " [";

  // TODO: Implement the printing of the conditional expression for the if.

  // ASTFile << "label=\"" << ConditionalNames;
  ASTFile << "label=\"" << this->getName();
  ASTFile << ", bb=" << printBBName(this->getCondExpr());
  ASTFile << "\"";
  ASTFile << ",shape=\"invhouse\",color=\"blue\"];\n";
}

void IfNode::dumpEdge(llvm::raw_fd_ostream &ASTFile) {
  if (this->getThen() != nullptr) {
    ASTFile << "node_" << this->getID() << " -> node_"
            << this->getThen()->getID() << " [color=green,label=\"then\"];\n";
  }

  if (this->getElse() != nullptr) {
    ASTFile << "node_" << this->getID() << " -> node_"
            << this->getElse()->getID() << " [color=green,label=\"else\"];\n";
  }
}

void ScsNode::dump(llvm::raw_fd_ostream &ASTFile) {
  ASTFile << "node_" << this->getID() << " [";
  ASTFile << "label=\"" << this->getName();
  if (this->RelatedCondition)
    ASTFile << ",bb=" << printBBName(this->RelatedCondition->getCondExpr());

  if (this->isWhileTrue())
    ASTFile << ",type=standard ";
  else if (this->isWhile())
    ASTFile << ",type=while ";
  if (this->isDoWhile())
    ASTFile << ",type=dowhile ";

  ASTFile << "\"";
  ASTFile << ",shape=\"circle\",color=\"black\"];\n";
}

void ScsNode::dumpEdge(llvm::raw_fd_ostream &ASTFile) {
  // After do-while and while match loop nodes could be empty
  // revng_assert(this->getBody() != nullptr);
  if (this->getBody() != nullptr) {
    ASTFile << "node_" << this->getID() << " -> node_"
            << this->getBody()->getID() << " [color=green,label=\"body\"];\n";
  }
}

void SequenceNode::dump(llvm::raw_fd_ostream &ASTFile) {
  ASTFile << "node_" << this->getID() << " [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"box\",color=\"black\"];\n";
}

void SequenceNode::dumpEdge(llvm::raw_fd_ostream &ASTFile) {
  int SuccessorIndex = 0;
  for (ASTNode *Successor : this->nodes()) {
    ASTFile << "node_" << this->getID() << " -> node_" << Successor->getID()
            << " [color=green,label=\"elem " << SuccessorIndex << "\"];\n";
    SuccessorIndex += 1;
  }
}

void SwitchNode::dump(llvm::raw_fd_ostream &ASTFile) {
  ASTFile << "node_" << this->getID() << " [";
  ASTFile << "label=\"" << this->getName();
  if (this->getOriginalBB() and not this->isWeaved())
    ASTFile << ",bb=" << this->getOriginalBB()->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"hexagon\",color=\"black\"];\n";
}

void SwitchNode::dumpEdge(llvm::raw_fd_ostream &ASTFile) {
  for (const auto &[LabelSet, Case] : cases()) {
    ASTFile << "node_" << this->getID() << " -> node_" << Case->getID()
            << " [color=green,label=\"case ";

    // Cases can now be sets of cases, we need to print all of them on a edge.
    for (uint64_t Label : LabelSet) {
      ASTFile << Label << ',';
    }

    // Close the line.
    ASTFile << "\"];\n";

    // Continue dumping the children of the switch node.
  }

  if (ASTNode *Default = this->getDefault()) {
    ASTFile << "node_" << this->getID() << " -> node_" << Default->getID()
            << " [color=green,label=\"default\"];\n";
  }
}

void BreakNode::dump(llvm::raw_fd_ostream &ASTFile) {
  ASTFile << "node_" << this->getID() << " [";
  ASTFile << "label=\"loop break\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void BreakNode::dumpEdge(llvm::raw_fd_ostream &ASTFile) {
}

void SwitchBreakNode::dump(llvm::raw_fd_ostream &ASTFile) {
  ASTFile << "node_" << this->getID() << " [";
  ASTFile << "label=\"switch break\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void SwitchBreakNode::dumpEdge(llvm::raw_fd_ostream &ASTFile) {
}

void ContinueNode::dump(llvm::raw_fd_ostream &ASTFile) {
  ASTFile << "node_" << this->getID() << " [";
  ASTFile << "label=\"continue\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void ContinueNode::dumpEdge(llvm::raw_fd_ostream &ASTFile) {
}

void SetNode::dump(llvm::raw_fd_ostream &ASTFile) {
  ASTFile << "node_" << this->getID() << " [";
  ASTFile << "label=\"" << this->getName();
  ASTFile << "\"";
  ASTFile << ",shape=\"box\",color=\"red\"];\n";
}

void SetNode::dumpEdge(llvm::raw_fd_ostream &ASTFile) {
}

void ASTNode::dump(llvm::raw_fd_ostream &ASTFile) {
  switch (getKind()) {
  case NK_Code:
    return llvm::cast<CodeNode>(this)->dump(ASTFile);
  case NK_Break:
    return llvm::cast<BreakNode>(this)->dump(ASTFile);
  case NK_Continue:
    return llvm::cast<ContinueNode>(this)->dump(ASTFile);
  case NK_If:
    return llvm::cast<IfNode>(this)->dump(ASTFile);
  case NK_Scs:
    return llvm::cast<ScsNode>(this)->dump(ASTFile);
  case NK_List:
    return llvm::cast<SequenceNode>(this)->dump(ASTFile);
  case NK_Switch:
    return llvm::cast<SwitchNode>(this)->dump(ASTFile);
  case NK_SwitchBreak:
    return llvm::cast<SwitchBreakNode>(this)->dump(ASTFile);
  case NK_Set:
    return llvm::cast<SetNode>(this)->dump(ASTFile);
  }
}

void ASTNode::dumpEdge(llvm::raw_fd_ostream &ASTFile) {
  switch (getKind()) {
  case NK_Code:
    return llvm::cast<CodeNode>(this)->dumpEdge(ASTFile);
  case NK_Break:
    return llvm::cast<BreakNode>(this)->dumpEdge(ASTFile);
  case NK_Continue:
    return llvm::cast<ContinueNode>(this)->dumpEdge(ASTFile);
  case NK_If:
    return llvm::cast<IfNode>(this)->dumpEdge(ASTFile);
  case NK_Scs:
    return llvm::cast<ScsNode>(this)->dumpEdge(ASTFile);
  case NK_List:
    return llvm::cast<SequenceNode>(this)->dumpEdge(ASTFile);
  case NK_Switch:
    return llvm::cast<SwitchNode>(this)->dumpEdge(ASTFile);
  case NK_SwitchBreak:
    return llvm::cast<SwitchBreakNode>(this)->dumpEdge(ASTFile);
  case NK_Set:
    return llvm::cast<SetNode>(this)->dumpEdge(ASTFile);
  }
}

void ASTNode::dumpSuccessor(llvm::raw_fd_ostream &ASTFile) {
  if (this->Successor != nullptr) {
    ASTFile << "node_" << this->getID() << " -> node_"
            << this->Successor->getID()
            << " [color=purple,label=\"successor\"];\n";
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
  case NodeKind::NK_If:
    delete static_cast<IfNode *>(A);
    break;
  case NodeKind::NK_Scs:
    delete static_cast<ScsNode *>(A);
    break;
  case NodeKind::NK_List:
    delete static_cast<SequenceNode *>(A);
    break;
  case NodeKind::NK_Switch:
    delete static_cast<SwitchNode *>(A);
    break;
  case NodeKind::NK_SwitchBreak:
    delete static_cast<SwitchBreakNode *>(A);
    break;
  case NodeKind::NK_Set:
    delete static_cast<SetNode *>(A);
    break;
  }
}
