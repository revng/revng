#ifndef REVNGC_RESTRUCTURE_CFG_ASTNODE_H
#define REVNGC_RESTRUCTURE_CFG_ASTNODE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>

// LLVM includes
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/Support/Casting.h>

// Local libraries includes
#include "revng-c/RestructureCFGPass/BasicBlockNodeBB.h"
#include "revng-c/RestructureCFGPass/ExprNode.h"

// Forward declarations
namespace llvm {
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
  // Steal the `BasicBlockNodeBB` definition from the external namespace
  using BasicBlockNodeBB = ::BasicBlockNodeBB;
  using BBNodeMap = std::map<BasicBlockNodeBB *, BasicBlockNodeBB *>;
  using ExprNodeMap = std::map<ExprNode *, ExprNode *>;

private:
  const NodeKind Kind;
  bool IsEmpty = false;

protected:
  llvm::BasicBlock *BB = nullptr;
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
    Name(Name),
    Successor(Successor) {}

  ASTNode(NodeKind K, BasicBlockNodeBB *CFGNode, ASTNode *Successor = nullptr) :
    Kind(K),
    IsEmpty(CFGNode->isEmpty()),
    BB(CFGNode->getOriginalNode()),
    Name(CFGNode->getNameStr()),
    Successor(Successor) {}

  inline ASTNode *Clone();

  ASTNode &operator=(ASTNode &&) = delete;
  ASTNode &operator=(const ASTNode &) = delete;
  ASTNode(ASTNode &&) = delete;
  ASTNode() = delete;

public:
  static void deleteASTNode(ASTNode *A);

protected:
  ASTNode(const ASTNode &) = default;
  ~ASTNode() = default;

public:
  NodeKind getKind() const { return Kind; }

  std::string getName() const {
    return "ID:" + std::to_string(getID()) + " Name:" + Name;
  }

  void setID(unsigned NewID) { ID = NewID; }

  unsigned getID() const { return ID; }

  llvm::BasicBlock *getBB() const { return BB; }

  ASTNode *getSuccessor() const { return Successor; }

  bool isEmpty() {

    // Since we do not have a pointer to the CFGNode anymore, we need to save
    // this information in a field inside the constructor.
    return IsEmpty;
  }

  llvm::BasicBlock *getOriginalBB() const { return BB; }

  inline void dump(std::ofstream &ASTFile);

  inline bool isEqual(const ASTNode *Node);

  inline void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);
};

class CodeNode : public ASTNode {
  friend class ASTNode;

public:
  CodeNode(BasicBlockNodeBB *CFGNode, ASTNode *Successor) :
    ASTNode(NK_Code, CFGNode, Successor) {}

protected:
  CodeNode(const CodeNode &) = default;
  CodeNode(CodeNode &&) = delete;
  ~CodeNode() = default;

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Code; }

  bool isEqual(const ASTNode *Node) const;

  void dump(std::ofstream &ASTFile);

  ASTNode *Clone() { return new CodeNode(*this); }
};

class IfNode : public ASTNode {
  friend class ASTNode;

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

protected:
  IfNode(const IfNode &) = default;
  IfNode(IfNode &&) = delete;
  ~IfNode() = default;

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

  bool isEqual(const ASTNode *Node) const;

  void dump(std::ofstream &ASTFile);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  ASTNode *Clone() { return new IfNode(*this); }

  ExprNode *getCondExpr() const { return ConditionExpression; }

  void replaceCondExpr(ExprNode *NewExpr) { ConditionExpression = NewExpr; }

  void updateCondExprPtr(ExprNodeMap &Map);
};

class ScsNode : public ASTNode {
  friend class ASTNode;

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

protected:
  ScsNode(const ScsNode &) = default;
  ScsNode(ScsNode &&) = delete;
  ~ScsNode() = default;

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Scs; }

  bool hasBody() const { return Body != nullptr; }

  ASTNode *getBody() const { return Body; }

  void setBody(ASTNode *Node) { Body = Node; }

  bool isEqual(const ASTNode *Node) const;

  void dump(std::ofstream &ASTFile);

  ASTNode *Clone() { return new ScsNode(*this); }

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
  friend class ASTNode;

public:
  using links_container = std::vector<ASTNode *>;
  using links_iterator = typename links_container::iterator;
  using links_range = llvm::iterator_range<links_iterator>;

private:
  links_container NodeList;

public:
  SequenceNode(std::string Name) : ASTNode(NK_List, Name) {}
  SequenceNode(BasicBlockNodeBB *CFGNode) : ASTNode(NK_List, CFGNode) {}

protected:
  SequenceNode(const SequenceNode &) = default;
  SequenceNode(SequenceNode &&) = delete;
  ~SequenceNode() = default;

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

  links_container::size_type listSize() const { return NodeList.size(); }

  ASTNode *getNodeN(links_container::size_type N) const { return NodeList[N]; }

  bool isEqual(const ASTNode *Node) const;

  void dump(std::ofstream &ASTFile);

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  ASTNode *Clone() {
    return reinterpret_cast<ASTNode *>(new SequenceNode(*this));
  }
};

class ContinueNode : public ASTNode {
  friend class ASTNode;

private:
  IfNode *ComputationIf = nullptr;
  bool IsImplicit = false;

public:
  ContinueNode() : ASTNode(NK_Continue, "continue") {}

protected:
  ContinueNode(const ContinueNode &) = default;
  ContinueNode(ContinueNode &&) = delete;
  ~ContinueNode() = default;

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Continue; }

  ASTNode *Clone() { return new ContinueNode(*this); }

  void dump(std::ofstream &ASTFile);

  bool isEqual(const ASTNode *Node) const {
    return nullptr != llvm::dyn_cast_or_null<ContinueNode>(Node);
  }

  bool hasComputation() const { return ComputationIf != nullptr; }

  void addComputationIfNode(IfNode *ComputationIfNode);

  IfNode *getComputationIfNode() const;

  bool isImplicit() const { return IsImplicit; }

  void setImplicit() { IsImplicit = true; }
};

class BreakNode : public ASTNode {
  friend class ASTNode;

public:
  BreakNode() : ASTNode(NK_Break, "loop break") {}

  static bool classof(const ASTNode *N) { return N->getKind() == NK_Break; }

protected:
  BreakNode(const BreakNode &) = default;
  BreakNode(BreakNode &&) = delete;
  ~BreakNode() = default;

public:
  ASTNode *Clone() { return new BreakNode(*this); }

  void dump(std::ofstream &ASTFile);

  bool isEqual(const ASTNode *Node) const {
    return nullptr != llvm::dyn_cast_or_null<BreakNode>(Node);
  }

  bool breaksFromWithinSwitch() const { return BreakFromWithinSwitch; }

  void setBreakFromWithinSwitch(bool B = true) { BreakFromWithinSwitch = B; }

protected:
  bool BreakFromWithinSwitch = false;
};

class SwitchBreakNode : public ASTNode {
  friend class ASTNode;

public:
  SwitchBreakNode() : ASTNode(NK_SwitchBreak, "switch break") {}

protected:
  SwitchBreakNode(const SwitchBreakNode &) = default;
  SwitchBreakNode(SwitchBreakNode &&) = delete;
  ~SwitchBreakNode() = default;

public:
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_SwitchBreak;
  }

  ASTNode *Clone() { return new SwitchBreakNode(*this); }

  void dump(std::ofstream &ASTFile);

  bool isEqual(const ASTNode *Node) const {
    return nullptr != llvm::dyn_cast_or_null<SwitchBreakNode>(Node);
  }
};

class SetNode : public ASTNode {
  friend class ASTNode;

private:
  unsigned StateVariableValue;

public:
  SetNode(BasicBlockNodeBB *CFGNode, ASTNode *Successor) :
    ASTNode(NK_Set, CFGNode, Successor),
    StateVariableValue(CFGNode->getStateVariableValue()) {}

  SetNode(BasicBlockNodeBB *CFGNode) :
    ASTNode(NK_Set, CFGNode, nullptr),
    StateVariableValue(CFGNode->getStateVariableValue()) {}

protected:
  SetNode(const SetNode &) = default;
  SetNode(SetNode &&) = delete;
  ~SetNode() = default;

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_Set; }

  bool isEqual(const ASTNode *Node) const;

  void dump(std::ofstream &ASTFile);

  ASTNode *Clone() { return new SetNode(*this); }

  unsigned getStateVariableValue() const { return StateVariableValue; }
};

class IfCheckNode : public IfNode {
  friend class ASTNode;

private:
  unsigned StateVariableValue;

public:
  IfCheckNode(BasicBlockNodeBB *CFGNode,
              ASTNode *Then,
              ASTNode *Else,
              ASTNode *PostDom) :
    IfNode(CFGNode, nullptr, Then, Else, PostDom, NK_IfCheck),
    StateVariableValue(CFGNode->getStateVariableValue()) {}

protected:
  IfCheckNode(const IfCheckNode &) = default;
  IfCheckNode(IfCheckNode &&) = delete;
  ~IfCheckNode() = default;

public:
  static bool classof(const ASTNode *N) { return N->getKind() == NK_IfCheck; }

  void dump(std::ofstream &ASTFile);

  ASTNode *Clone() { return new IfCheckNode(*this); }

  unsigned getCaseValue() const { return StateVariableValue; }
};

// Abstract SwitchNode. It has the concept of cases (other ASTNodes) but no
// concept of values for which those cases are activated.
class SwitchNode : public ASTNode {
  friend class ASTNode;

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

  SwitchNode(NodeKind K,
             case_container &Cases,
             const std::string &Name,
             ASTNode *Def = nullptr,
             ASTNode *Successor = nullptr) :
    ASTNode(K, Name, Successor),
    CaseVec(Cases),
    Default(Def) {}

  SwitchNode(const SwitchNode &) = default;
  SwitchNode(SwitchNode &&) = delete;
  ~SwitchNode() = default;

public:
  static bool classof(const ASTNode *N) {
    return N->getKind() >= NK_SwitchRegular and N->getKind() <= NK_SwitchCheck;
  }

  case_range unordered_cases() {
    return llvm::make_range(CaseVec.begin(), CaseVec.end());
  }

  ASTNode *getCaseN(case_container::size_type N) const {
    revng_assert(N < CaseSize());
    return CaseVec[N];
  }

  size_t CaseSize() const { return CaseVec.size(); }

  void updateASTNodesPointers(ASTNodeMap &SubstitutionMap);

  bool needsStateVariable() const { return NeedStateVariable; }

  void setNeedsStateVariable(bool N = true) { NeedStateVariable = N; }

  bool needsLoopBreakDispatcher() const { return NeedLoopBreakDispatcher; }

  void setNeedsLoopBreakDispatcher(bool N = true) {
    NeedLoopBreakDispatcher = N;
  }

  ASTNode *getDefault() const { return Default; }

  bool isEqual(const ASTNode *Node) const;

protected:
  bool hasEqualCaseValues(const SwitchNode *Node) const;
  case_container CaseVec;
  ASTNode *const Default;
  bool NeedStateVariable = false; // for breaking directly out of a loop
  bool NeedLoopBreakDispatcher = false; // to dispatchg breaks out of a loop
};

class RegularSwitchNode : public SwitchNode {
  friend class ASTNode;

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

  // Constructor for switch construction during AST generation phase.
  RegularSwitchNode(llvm::Value *Cond,
                    case_container &Cases,
                    case_value_container &V,
                    ASTNode *Def = nullptr,
                    ASTNode *Successor = nullptr) :
    SwitchNode(NK_SwitchRegular, Cases, "SwitchNode", Def, Successor),
    Condition(Cond),
    CaseValueVec(V) {
    revng_assert(Cases.size() == V.size());
  }

protected:
  RegularSwitchNode(const RegularSwitchNode &) = default;
  RegularSwitchNode(RegularSwitchNode &&) = delete;
  ~RegularSwitchNode() = default;

public:
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_SwitchRegular;
  }

public:
  void dump(std::ofstream &ASTFile);

  ASTNode *Clone() { return new RegularSwitchNode(*this); }

  llvm::Value *getCondition() const { return Condition; }

  case_value getCaseValueN(case_container::size_type N) const {
    revng_assert(N < CaseSize());
    return CaseValueVec[N];
  }

protected:
  llvm::Value *const Condition;
  const case_value_container CaseValueVec;
};

class SwitchCheckNode : public SwitchNode {
  friend class ASTNode;

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

  SwitchCheckNode(case_container &Cases,
                  case_value_container &V,
                  ASTNode *Def = nullptr,
                  ASTNode *Successor = nullptr) :
    SwitchNode(NK_SwitchCheck, Cases, "SwitchCheckNode", Def, Successor),
    CaseValueVec(V) {
    revng_assert(Cases.size() == V.size());
  }

protected:
  SwitchCheckNode(const SwitchCheckNode &) = default;
  SwitchCheckNode(SwitchCheckNode &&) = delete;
  ~SwitchCheckNode() = default;

public:
  static bool classof(const ASTNode *N) {
    return N->getKind() == NK_SwitchCheck;
  }

public:
  void dump(std::ofstream &ASTFile);

  case_value getCaseValueN(size_t N) const {
    revng_assert(N < CaseSize());
    return CaseValueVec[N];
  }

  ASTNode *Clone() { return new SwitchCheckNode(*this); }

protected:
  const case_value_container CaseValueVec;
};

inline ASTNode *ASTNode::Clone() {
  switch (getKind()) {
  case NK_Code:
    return llvm::cast<CodeNode>(this)->Clone();
  case NK_Break:
    return llvm::cast<BreakNode>(this)->Clone();
  case NK_Continue:
    return llvm::cast<ContinueNode>(this)->Clone();
  // ---- IfNode kinds
  case NK_If:
    return llvm::cast<IfNode>(this)->Clone();
  case NK_IfCheck:
    return llvm::cast<IfCheckNode>(this)->Clone();
  // ---- end IfNode kinds
  case NK_Scs:
    return llvm::cast<ScsNode>(this)->Clone();
  case NK_List:
    return llvm::cast<SequenceNode>(this)->Clone();
  // ---- SwitchNode kinds
  case NK_SwitchRegular:
    return llvm::cast<RegularSwitchNode>(this)->Clone();
  case NK_SwitchCheck:
    return llvm::cast<SwitchCheckNode>(this)->Clone();
  // ---- end SwitchNode kinds
  case NK_SwitchBreak:
    return llvm::cast<SwitchBreakNode>(this)->Clone();
  case NK_Set:
    return llvm::cast<SetNode>(this)->Clone();
  }
  return nullptr;
}

inline void ASTNode::updateASTNodesPointers(ASTNodeMap &SubstitutionMap) {
  if (IfNode *If = llvm::dyn_cast<IfNode>(this)) {
    If->updateASTNodesPointers(SubstitutionMap);
  } else if (SequenceNode *Seq = llvm::dyn_cast<SequenceNode>(this)) {
    Seq->updateASTNodesPointers(SubstitutionMap);
  } else if (SwitchNode *Switch = llvm::dyn_cast<SwitchNode>(this)) {
    Switch->updateASTNodesPointers(SubstitutionMap);
  }
}

inline void ASTNode::dump(std::ofstream &ASTFile) {
  switch (getKind()) {
  case NK_Code:
    return llvm::cast<CodeNode>(this)->dump(ASTFile);
  case NK_Break:
    return llvm::cast<BreakNode>(this)->dump(ASTFile);
  case NK_Continue:
    return llvm::cast<ContinueNode>(this)->dump(ASTFile);
  // ---- IfNode kinds
  case NK_If:
    return llvm::cast<IfNode>(this)->dump(ASTFile);
  case NK_IfCheck:
    return llvm::cast<IfCheckNode>(this)->dump(ASTFile);
  // ---- end IfNode kinds
  case NK_Scs:
    return llvm::cast<ScsNode>(this)->dump(ASTFile);
  case NK_List:
    return llvm::cast<SequenceNode>(this)->dump(ASTFile);
  // ---- SwitchNode kinds
  case NK_SwitchRegular:
    return llvm::cast<RegularSwitchNode>(this)->dump(ASTFile);
  case NK_SwitchCheck:
    return llvm::cast<SwitchCheckNode>(this)->dump(ASTFile);
  // ---- end SwitchNode kinds
  case NK_SwitchBreak:
    return llvm::cast<SwitchBreakNode>(this)->dump(ASTFile);
  case NK_Set:
    return llvm::cast<SetNode>(this)->dump(ASTFile);
  }
}

inline bool ASTNode::isEqual(const ASTNode *Node) {
  switch (getKind()) {
  case NK_Code:
    return llvm::cast<CodeNode>(this)->isEqual(Node);
  case NK_Break:
    return llvm::cast<BreakNode>(this)->isEqual(Node);
  case NK_Continue:
    return llvm::cast<ContinueNode>(this)->isEqual(Node);
  // ---- IfNode kinds
  case NK_If:
    return llvm::cast<IfNode>(this)->isEqual(Node);
  case NK_IfCheck:
    return llvm::cast<IfCheckNode>(this)->isEqual(Node);
  // ---- end IfNode kinds
  case NK_Scs:
    return llvm::cast<ScsNode>(this)->isEqual(Node);
  case NK_List:
    return llvm::cast<SequenceNode>(this)->isEqual(Node);
  // ---- SwitchNode kinds
  case NK_SwitchRegular:
    return llvm::cast<SwitchNode>(this)->isEqual(Node);
  case NK_SwitchCheck:
    return llvm::cast<SwitchCheckNode>(this)->isEqual(Node);
  // ---- end SwitchNode kinds
  case NK_SwitchBreak:
    return llvm::cast<SwitchBreakNode>(this)->isEqual(Node);
  case NK_Set:
    return llvm::cast<SetNode>(this)->isEqual(Node);
  default:
    revng_abort();
  }
}
#endif // define REVNGC_RESTRUCTURE_CFG_ASTNODE_H
