#ifndef REVNGC_RESTRUCTURE_CFG_ASTNODE_H
#define REVNGC_RESTRUCTURE_CFG_ASTNODE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>

// LLVM includes
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

// local libraries includes
#include "revng-c/RestructureCFGPass/BasicBlockNode.h"
#include "revng-c/RestructureCFGPass/ExprNode.h"

// forward declarations
namespace llvm {
class BasicBlock;
class ConstantInt;
} // namespace llvm

template<class NodeT>
class BasicBlockNode;

class ASTNode {

public:
  enum NodeKind {
    NK_Code,
    NK_Break,
    NK_Continue,
    // ---- IfNode kinds
    NK_If,
    NK_IfCheck,
    // ---- end IfNode kinds
    NK_Scs,
    NK_List,
    // ---- SwitchNode kinds
    NK_SwitchRegular,
    NK_SwitchCheck,
    // ---- end SwitchNode kinds
    NK_SwitchBreak,
    NK_Set
  };

  using ASTNodeMap = std::map<ASTNode *, ASTNode *>;
  using BasicBlockNodeBB = BasicBlockNode<llvm::BasicBlock *>;
  using BBNodeMap = std::map<BasicBlockNodeBB *, BasicBlockNodeBB *>;
  using ExprNodeMap = std::map<ExprNode *, ExprNode *>;

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

  ASTNode(NodeKind K, BasicBlockNodeBB *CFGNode, ASTNode *Successor = nullptr) :
    Kind(K),
    BB(CFGNode->getOriginalNode()),
    Name(CFGNode->getNameStr()),
    Successor(Successor),
    IsEmpty(CFGNode->isEmpty()) {}

  virtual ~ASTNode() = default;

  virtual ASTNode *Clone() = 0;

public:
  NodeKind getKind() const { return Kind; }

  std::string getName() const {
    return "ID:" + std::to_string(getID()) + " Name:" + Name;
  }

  void setID(unsigned NewID) { ID = NewID; }

  unsigned getID() const { return ID; }

  virtual void dump(std::ofstream &ASTFile) = 0;

  llvm::BasicBlock *getBB() const { return BB; }

  ASTNode *getSuccessor() const { return Successor; }

  bool isEmpty() {

    // Since we do not have a pointer to the CFGNode anymore, we need to save
    // this information in a field inside the constructor.
    return IsEmpty;
  }

  llvm::BasicBlock *getOriginalBB() const { return BB; }

  virtual bool isEqual(const ASTNode *Node) const = 0;

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) = 0;
};

class CodeNode : public ASTNode {

public:
  CodeNode(BasicBlockNodeBB *CFGNode, ASTNode *Successor) :
    ASTNode(NK_Code, CFGNode, Successor) {}

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Code; }

  virtual bool isEqual(const ASTNode *Node) const override;

  virtual void dump(std::ofstream &ASTFile) override;

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) override{};

  virtual ASTNode *Clone() override { return new CodeNode(*this); }

  virtual ~CodeNode() override = default;
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
  IfNode(BasicBlockNodeBB *CFGNode,
         ExprNode *CondExpr,
         ASTNode *Then,
         ASTNode *Else,
         ASTNode *PostDom,
         NodeKind Kind = NK_If) :
    ASTNode(Kind, CFGNode, PostDom),
    Then(Then),
    Else(Else),
    ConditionExpression(CondExpr) {}

public:
  static bool classof(const ASTNode *N) {
    return N->getKind() >= NK_If && N->getKind() <= NK_IfCheck;
  }

  ASTNode *getThen() const { return Then; }

  ASTNode *getElse() const { return Else; }

  void setThen(ASTNode *Node) { Then = Node; }

  void setElse(ASTNode *Node) { Else = Node; }

  bool hasThen() const {
    if (Then != nullptr) {
      return true;
    }
    return false;
  }

  bool hasElse() const {
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

  virtual bool isEqual(const ASTNode *Node) const override;

  virtual void dump(std::ofstream &ASTFile) override;

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) override;

  virtual ASTNode *Clone() override { return new IfNode(*this); }

  virtual ~IfNode() override = default;

  ExprNode *getCondExpr() const { return ConditionExpression; }

  void replaceCondExpr(ExprNode *NewExpr) { ConditionExpression = NewExpr; }

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
  ScsNode(BasicBlockNodeBB *CFGNode, ASTNode *Body) :
    ASTNode(NK_Scs, CFGNode, nullptr),
    Body(Body) {}

  ScsNode(BasicBlockNodeBB *CFGNode, ASTNode *Body, ASTNode *Successor) :
    ASTNode(NK_Scs, CFGNode, Successor),
    Body(Body) {}

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Scs; }

  bool hasBody() const { return Body != nullptr; }

  ASTNode *getBody() const { return Body; }

  void setBody(ASTNode *Node) { Body = Node; }

  virtual bool isEqual(const ASTNode *Node) const override;

  virtual void dump(std::ofstream &ASTFile) override;

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) override{};

  virtual ASTNode *Clone() override { return new ScsNode(*this); }

  virtual ~ScsNode() override = default;

  bool isStandard() const { return LoopType == Type::Standard; }

  bool isWhile() const { return LoopType == Type::While; }

  bool isDoWhile() const { return LoopType == Type::DoWhile; }

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
  SequenceNode(BasicBlockNodeBB *CFGNode) : ASTNode(NK_List, CFGNode) {}

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

  int listSize() const { return NodeList.size(); }

  ASTNode *getNodeN(int N) const { return NodeList[N]; }

  virtual bool isEqual(const ASTNode *Node) const override;

  virtual void dump(std::ofstream &ASTFile) override;

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) override;

  virtual ASTNode *Clone() override { return new SequenceNode(*this); }

  virtual ~SequenceNode() override = default;
};

class ContinueNode : public ASTNode {
private:
  IfNode *ComputationIf = nullptr;
  bool IsImplicit = false;

public:
  ContinueNode() : ASTNode(NK_Continue, "continue"){};

  static bool classof(const ASTNode *N) { return N->getKind() == NK_Continue; }

  virtual ASTNode *Clone() override { return new ContinueNode(*this); }

  virtual void dump(std::ofstream &ASTFile) override;

  virtual bool isEqual(const ASTNode *Node) const override {
    return nullptr != llvm::dyn_cast_or_null<ContinueNode>(Node);
  }

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) override {}

  virtual ~ContinueNode() override = default;

  bool hasComputation() const { return ComputationIf != nullptr; };

  void addComputationIfNode(IfNode *ComputationIfNode);

  IfNode *getComputationIfNode() const;

  bool isImplicit() const { return IsImplicit; };

  void setImplicit() { IsImplicit = true; };
};

class BreakNode : public ASTNode {

public:
  BreakNode() : ASTNode(NK_Break, "loop break"){};

  static bool classof(const ASTNode *N) { return N->getKind() == NK_Break; }

  virtual ASTNode *Clone() override { return new BreakNode(*this); }

  virtual void dump(std::ofstream &ASTFile) override;

  virtual bool isEqual(const ASTNode *Node) const override {
    return nullptr != llvm::dyn_cast_or_null<BreakNode>(Node);
  }

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) override {}

  virtual ~BreakNode() override = default;

  bool breaksFromWithinSwitch() const { return BreakFromWithinSwitch; }

  void setBreakFromWithinSwitch(bool B = true) { BreakFromWithinSwitch = B; }

protected:
  bool BreakFromWithinSwitch = false;
};

class SwitchBreakNode : public ASTNode {

public:
  SwitchBreakNode() : ASTNode(NK_SwitchBreak, "switch break"){};

  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_SwitchBreak;
  }

  virtual ASTNode *Clone() override { return new SwitchBreakNode(*this); }

  virtual void dump(std::ofstream &ASTFile) override;

  virtual bool isEqual(const ASTNode *Node) const override {
    return nullptr != llvm::dyn_cast_or_null<SwitchBreakNode>(Node);
  }

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) override {}

  virtual ~SwitchBreakNode() override = default;
};

class SetNode : public ASTNode {

private:
  unsigned StateVariableValue;

public:
  SetNode(BasicBlockNodeBB *CFGNode, ASTNode *Successor) :
    ASTNode(NK_Set, CFGNode, Successor),
    StateVariableValue(CFGNode->getStateVariableValue()) {}

  SetNode(BasicBlockNodeBB *CFGNode) :
    ASTNode(NK_Set, CFGNode, nullptr),
    StateVariableValue(CFGNode->getStateVariableValue()) {}

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Set; }

  virtual bool isEqual(const ASTNode *Node) const override;

  virtual void dump(std::ofstream &ASTFile) override;

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) override{};

  virtual ASTNode *Clone() override { return new SetNode(*this); }

  virtual ~SetNode() override = default;

  unsigned getStateVariableValue() const { return StateVariableValue; }
};

class IfCheckNode : public IfNode {

private:
  unsigned StateVariableValue;

public:
  IfCheckNode(BasicBlockNodeBB *CFGNode,
              ASTNode *Then,
              ASTNode *Else,
              ASTNode *PostDom) :
    IfNode(CFGNode, nullptr, Then, Else, PostDom, NK_IfCheck),
    StateVariableValue(CFGNode->getStateVariableValue()) {}

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_IfCheck; }

  virtual void dump(std::ofstream &ASTFile) override;

  virtual ASTNode *Clone() override { return new IfCheckNode(*this); }

  unsigned getCaseValue() const { return StateVariableValue; }
};

// Abstract SwitchNode. It has the concept of cases (other ASTNodes) but no
// concept of values for which those cases are activated.
class SwitchNode : public ASTNode {
protected:
  static const constexpr int SwitchNumCases = 16;

public:
  using case_container = llvm::SmallVector<ASTNode *, SwitchNumCases>;
  using case_iterator = typename case_container::iterator;
  using case_range = llvm::iterator_range<case_iterator>;

protected:
  SwitchNode(NodeKind K,
             const case_container &Cases,
             const std::string &Name,
             ASTNode *Def = nullptr) :
    ASTNode(K, Name),
    CaseVec(Cases),
    Default(Def) {}

  SwitchNode(NodeKind K,
             case_container &&Cases,
             const std::string &Name,
             ASTNode *Def = nullptr) :
    ASTNode(K, Name),
    CaseVec(Cases),
    Default(Def) {}

public:
  ~SwitchNode() override = default;

  static bool classof(const ASTNode *N) {
    return N->getKind() >= NK_SwitchRegular and N->getKind() <= NK_SwitchCheck;
  }

  case_range unordered_cases() {
    return llvm::make_range(CaseVec.begin(), CaseVec.end());
  }

  virtual ASTNode *getCaseN(int N) const {
    revng_assert((-1 < N) and (N < CaseSize()));
    return CaseVec[N];
  }

  size_t CaseSize() const { return CaseVec.size(); }

  virtual void updateASTNodesPointers(ASTNodeMap &SubstitutionMap) override;

  bool needsStateVariable() const { return NeedStateVariable; }

  void setNeedsStateVariable(bool N = true) { NeedStateVariable = N; }

  bool needsLoopBreakDispatcher() const { return NeedLoopBreakDispatcher; }

  void setNeedsLoopBreakDispatcher(bool N = true) {
    NeedLoopBreakDispatcher = N;
  }

  ASTNode *getDefault() const { return Default; }

  virtual bool isEqual(const ASTNode *Node) const override;

protected:
  bool hasEqualCaseValues(const SwitchNode *Node) const;
  case_container CaseVec;
  ASTNode *const Default;
  bool NeedStateVariable = false; // for breaking directly out of a loop
  bool NeedLoopBreakDispatcher = false; // to dispatchg breaks out of a loop
};

class RegularSwitchNode : public SwitchNode {

public:
  using case_value = llvm::ConstantInt *;
  using case_value_container = llvm::SmallVector<case_value, SwitchNumCases>;
  using case_value_iterator = typename case_value_container::iterator;
  using case_value_range = llvm::iterator_range<case_value_iterator>;

public:
  RegularSwitchNode(llvm::Value *Cond,
                    const case_container &Cases,
                    const case_value_container &V,
                    ASTNode *Def = nullptr) :
    SwitchNode(NK_SwitchRegular, Cases, "SwitchNode", Def),
    Condition(Cond),
    CaseValueVec(V) {
    revng_assert(Cases.size() == V.size());
  }

  RegularSwitchNode(llvm::Value *Cond,
                    case_container &&Cases,
                    const case_value_container &V,
                    ASTNode *Def = nullptr) :
    SwitchNode(NK_SwitchRegular, Cases, "SwitchNode", Def),
    Condition(Cond),
    CaseValueVec(V) {
    revng_assert(Cases.size() == V.size());
  }

  RegularSwitchNode(llvm::Value *Cond,
                    const case_container &Cases,
                    case_value_container &&V,
                    ASTNode *Def = nullptr) :
    SwitchNode(NK_SwitchRegular, Cases, "SwitchNode", Def),
    Condition(Cond),
    CaseValueVec(V) {
    revng_assert(Cases.size() == V.size());
  }

  RegularSwitchNode(llvm::Value *Cond,
                    case_container &&Cases,
                    case_value_container &&V,
                    ASTNode *Def = nullptr) :
    SwitchNode(NK_SwitchRegular, Cases, "SwitchNode", Def),
    Condition(Cond),
    CaseValueVec(V) {
    revng_assert(Cases.size() == V.size());
  }

  virtual ~RegularSwitchNode() override = default;

  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_SwitchRegular;
  }

  virtual void dump(std::ofstream &ASTFile) override;

  virtual ASTNode *Clone() override { return new RegularSwitchNode(*this); }

  llvm::Value *getCondition() const { return Condition; }

  case_value getCaseValueN(int N) const {
    revng_assert((-1 < N) and (N < CaseSize()));
    return CaseValueVec[N];
  }

protected:
  const case_value_container CaseValueVec;
  llvm::Value *const Condition;
};

class SwitchCheckNode : public SwitchNode {
public:
  using case_value = uint64_t;
  using case_value_container = llvm::SmallVector<case_value, SwitchNumCases>;
  using case_value_iterator = typename case_value_container::iterator;
  using case_value_range = llvm::iterator_range<case_value_iterator>;

public:
  SwitchCheckNode(const case_container &Cases,
                  const case_value_container &V,
                  ASTNode *Def = nullptr) :
    SwitchNode(NK_SwitchCheck, Cases, "SwitchCheckNode", Def),
    CaseValueVec(V) {
    revng_assert(Cases.size() == V.size());
  }

  SwitchCheckNode(const case_container &Cases,
                  case_value_container &V,
                  ASTNode *Def = nullptr) :
    SwitchNode(NK_SwitchCheck, Cases, "SwitchCheckNode", Def),
    CaseValueVec(V) {
    revng_assert(Cases.size() == V.size());
  }

  SwitchCheckNode(case_container &&Cases,
                  const case_value_container &&V,
                  ASTNode *Def = nullptr) :
    SwitchNode(NK_SwitchCheck, Cases, "SwitchCheckNode", Def),
    CaseValueVec(V) {
    revng_assert(Cases.size() == V.size());
  }

  SwitchCheckNode(case_container &&Cases,
                  case_value_container &&V,
                  ASTNode *Def = nullptr) :
    SwitchNode(NK_SwitchCheck, Cases, "SwitchCheckNode", Def),
    CaseValueVec(V) {
    revng_assert(Cases.size() == V.size());
  }

  virtual ~SwitchCheckNode() override = default;

  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_SwitchCheck;
  }

  virtual void dump(std::ofstream &ASTFile) override;

  virtual ASTNode *Clone() override { return new SwitchCheckNode(*this); }

  case_value getCaseValueN(int N) const {
    revng_assert((-1 < N) and (N < CaseSize()));
    return CaseValueVec[N];
  }

protected:
  const case_value_container CaseValueVec;
};
#endif // define REVNGC_RESTRUCTURE_CFG_ASTNODE_H
