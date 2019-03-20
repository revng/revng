#ifndef REVNGC_RESTRUCTURE_CFG_ASTNODE_H
#define REVNGC_RESTRUCTURE_CFG_ASTNODE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>

// LLVM includes
#include <llvm/Support/Casting.h>

// local libraries includes
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
#include "revng-c/RestructureCFGPass/ExprNode.h"

// forward declarations
namespace llvm {
class BasicBlock;
class ConstantInt;
}

class BasicBlockNode;

using BBNodeMap = std::map<BasicBlockNode *, BasicBlockNode *>;
using ExprNodeMap = std::map<ExprNode *, ExprNode *>;

class ASTNode {

public:
  enum NodeKind { NK_Code,
                  NK_Break,
                  NK_Continue,
                  NK_If,
                  NK_IfCheck,
                  NK_Scs,
                  NK_List,
                  NK_Switch,
                  NK_Set };

  using ASTNodeMap = std::map<ASTNode *, ASTNode *>;

private:
  const NodeKind Kind;
  bool IsEmpty = false;

protected:
  llvm::BasicBlock *BB;
  bool Processed = false;
  std::string Name;
  ASTNode *Successor = nullptr;

  /// Unique Node ID inside a ASTNode, useful for printing to graphviz
  /// This field is initialized to 0, and will be re-assigned when the ASTNode
  /// will be inserted in an ASTTree.
  unsigned ID = 0;

public:
  ASTNode(NodeKind K, const std::string &Name, ASTNode *Successor = nullptr) :
    Kind(K),
    BB(nullptr),
    Name(Name),
    Successor(Successor) {}

  ASTNode(NodeKind K, BasicBlockNode *CFGNode, ASTNode *Successor = nullptr) :
    Kind(K),
    BB(CFGNode->getBasicBlock()),
    Name(CFGNode->getNameStr()),
    Successor(Successor),
    IsEmpty(CFGNode->isEmpty()) {}

  virtual ~ASTNode() {}

  virtual ASTNode *Clone() = 0;

public:
  NodeKind getKind() const { return Kind; }

  std::string getName() {
    return "ID:" + std::to_string(getID()) + " Name:" + Name;
  }

  void setID(unsigned NewID) { ID = NewID; }

  unsigned getID() const { return ID; }

  virtual void dump(std::ofstream &ASTFile) = 0;

  llvm::BasicBlock *getBB() { return BB; }

  ASTNode *getSuccessor() { return Successor; }

  bool isEmpty() {

    // Since we do not have a pointer to the CFGNode anymore, we need to save
    // this information in a field inside the constructor.
    return IsEmpty;
  }

  llvm::BasicBlock *getOriginalBB() {
    return BB;
  }

  virtual bool isEqual(ASTNode *Node) = 0;

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) = 0;
};

class CodeNode : public ASTNode {

public:
  CodeNode(BasicBlockNode *CFGNode, ASTNode *Successor) :
    ASTNode(NK_Code, CFGNode, Successor) {}

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Code; }

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  ASTNode *Clone() { return new CodeNode(*this); }
};

class IfNode : public ASTNode {

public:
  using links_container = std::vector<llvm::BasicBlock *>;
  using links_iterator = typename links_container::iterator;
  using links_range = llvm::iterator_range<links_iterator>;

protected:
  ASTNode *Then;
  ASTNode *Else;
  ExprNode *ConditionExpression;

public:
  IfNode(BasicBlockNode *CFGNode,
         ExprNode *CondExpr,
         ASTNode *Then,
         ASTNode *Else,
         ASTNode *PostDom,
         NodeKind Kind = NK_If) :
    ASTNode(Kind, CFGNode, PostDom),
    Then(Then),
    Else(Else),
    ConditionExpression(CondExpr) {
  }

public:
  static bool classof(const ASTNode *N) { return N->getKind() >= NK_If &&
                                                 N->getKind() <= NK_IfCheck; }

  ASTNode *getThen() { return Then; }

  ASTNode *getElse() { return Else; }

  void setThen(ASTNode *Node) { Then = Node; }

  void setElse(ASTNode *Node) { Else = Node; }

  bool hasThen() {
    if (Then != nullptr) {
      return true;
    }
    return false;
  }

  bool hasElse() {
    if (Else != nullptr) {
      return true;
    }
    return false;
  }

  bool hasBothBranches() {
    if ((Then != nullptr) and (Else != nullptr)) {
      return true;
    } else {
      return false;
    }
  }

  bool isEqual(ASTNode *Node);

  virtual void dump(std::ofstream &ASTFile);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  virtual ASTNode *Clone() { return new IfNode(*this); }

  ExprNode *getCondExpr() {
    return ConditionExpression;
  }

  void replaceCondExpr(ExprNode *NewExpr) {
    ConditionExpression = NewExpr;
  }

  void updateCondExprPtr(ExprNodeMap &Map);
};

class ScsNode : public ASTNode {

public:
  enum class Type {
    Standard,
    While,
    DoWhile,
  };

private:
  ASTNode *Body;
  Type LoopType = Type::Standard;
  IfNode *RelatedCondition = nullptr;

public:
  ScsNode(BasicBlockNode *CFGNode, ASTNode *Body) :
    ASTNode(NK_Scs, CFGNode),
    Body(Body) {}
  ScsNode(BasicBlockNode *CFGNode, ASTNode *Body, ASTNode *Successor) :
    ASTNode(NK_Scs, CFGNode, Successor),
    Body(Body) {}

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Scs; }

  ASTNode *getBody() { return Body; }

  void setBody(ASTNode *Node) { Body = Node; }

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  ASTNode *Clone() { return new ScsNode(*this); }

  bool isStandard() { return LoopType == Type::Standard; }

  bool isWhile() { return LoopType == Type::While; }

  bool isDoWhile() { return LoopType == Type::DoWhile; }

  void setWhile(IfNode *Condition) {
    LoopType = Type::While;
    RelatedCondition = Condition;
  }

  void setDoWhile(IfNode *Condition) {
    LoopType = Type::DoWhile;
    RelatedCondition = Condition;
  }

  IfNode *getRelatedCondition() {
    revng_assert(LoopType == Type::While or LoopType == Type::DoWhile);
    revng_assert(RelatedCondition != nullptr);

    return RelatedCondition;
  }
};

class SequenceNode : public ASTNode {

public:
  using links_container = std::vector<ASTNode *>;
  using links_iterator = typename links_container::iterator;
  using links_range = llvm::iterator_range<links_iterator>;

private:
  links_container NodeList;

public:
  SequenceNode(std::string Name) : ASTNode(NK_List, Name) {}
  SequenceNode(BasicBlockNode *CFGNode) : ASTNode(NK_List, CFGNode) {}

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_List; }

  links_range nodes() {
    return llvm::make_range(NodeList.begin(), NodeList.end());
  }

  void addNode(ASTNode *Node) {
    NodeList.push_back(Node);
    if (Node->getSuccessor() != nullptr) {
      this->addNode(Node->getSuccessor());
    }
  }

  void removeNode(ASTNode *Node) {
    NodeList.erase(std::remove(NodeList.begin(), NodeList.end(), Node),
                   NodeList.end());
  }

  int listSize() { return NodeList.size(); }

  ASTNode *getNodeN(int N) { return NodeList[N]; }

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  ASTNode *Clone() { return new SequenceNode(*this); }
};

class ContinueNode : public ASTNode {
private:
  IfNode *ComputationIf = nullptr;

public:
  ContinueNode() : ASTNode(NK_Continue, "continue"){};

  static bool classof(const ASTNode *N) { return N->getKind() == NK_Continue; }

  ASTNode *Clone() { return new ContinueNode(*this); }

  void dump(std::ofstream &ASTFile);

  bool isEqual(ASTNode *Node) { return llvm::isa<ContinueNode>(Node); }

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {}

  bool hasComputation() { return ComputationIf != nullptr; };

  void addComputationIfNode(IfNode *ComputationIfNode);

  IfNode *getComputationIfNode();
};

class BreakNode : public ASTNode {

public:
  BreakNode() : ASTNode(NK_Break, "break"){};

  static bool classof(const ASTNode *N) { return N->getKind() == NK_Break; }

  ASTNode *Clone() { return new BreakNode(*this); }

  void dump(std::ofstream &ASTFile);

  bool isEqual(ASTNode *Node) { return llvm::isa<BreakNode>(Node); }

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {}
};

class SwitchNode : public ASTNode {

public:
  using links_container = std::vector<std::pair<llvm::ConstantInt *, ASTNode *>>;
  using links_iterator = typename links_container::iterator;
  using links_range = llvm::iterator_range<links_iterator>;

private:
  links_container CaseList;
  llvm::Value *SwitchCondition;

public:
  SwitchNode(llvm::Value *Condition,
             std::vector<std::pair<llvm::ConstantInt *, ASTNode *>> &Cases) :
    ASTNode(NK_Switch, "SwitchNode"), SwitchCondition(Condition) {
      for(auto &Case : Cases) {
        CaseList.push_back(Case);
      }
    }

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Switch; }

  links_range cases() {
    return llvm::make_range(CaseList.begin(), CaseList.end());
  }

  int CaseSize() { return CaseList.size(); }

  ASTNode *getCaseN(int N) { return CaseList[N].second; }

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  ASTNode *Clone() { return new SwitchNode(*this); }

  llvm::Value *getCondition() { return SwitchCondition; }
};

class SetNode : public ASTNode {

private:
  unsigned StateVariableValue;

public:
  SetNode(BasicBlockNode *CFGNode, ASTNode *Successor) :
    ASTNode(NK_Set, CFGNode, Successor),
    StateVariableValue(CFGNode->getStateVariableValue()) {}

  SetNode(BasicBlockNode *CFGNode) :
    ASTNode(NK_Set, CFGNode),
    StateVariableValue(CFGNode->getStateVariableValue()) {}

public:
  static bool classof(const ASTNode *N) {return N->getKind() == NK_Set; }

  bool isEqual(ASTNode *Node);

  void dump(std::ofstream &ASTFile);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  ASTNode *Clone() {return new SetNode(*this); }

  unsigned getStateVariableValue() { return StateVariableValue; }
};

class IfCheckNode : public IfNode {

private:
  unsigned StateVariableValue;

public:
  IfCheckNode(BasicBlockNode *CFGNode,
              ASTNode *Then,
              ASTNode *Else,
              ASTNode *PostDom) :
    IfNode(CFGNode, nullptr, Then, Else, PostDom, NK_IfCheck),
    StateVariableValue(CFGNode->getStateVariableValue()) {
  }

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_IfCheck; }

  void dump(std::ofstream &ASTFile);

  ASTNode *Clone() { return new IfCheckNode(*this); }

  unsigned getCaseValue() { return StateVariableValue; }

};

#endif // define REVNGC_RESTRUCTURE_CFG_ASTNODE_H
