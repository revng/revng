#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <cstdlib>

#include "llvm/Support/Casting.h"

class AtomicNode;
class NotNode;
class AndNode;
class OrNode;

namespace llvm {
class BasicBlock;
class Value;
} // namespace llvm

class ExprNode {
public:
  enum NodeKind {
    NK_ValueCompare,
    NK_LoopStateCompare,
    NK_Atomic,
    NK_Not,
    NK_And,
    NK_Or
  };

protected:
  const NodeKind Kind;

public:
  NodeKind getKind() const { return Kind; }

  ExprNode(const ExprNode &) = default;
  ExprNode(ExprNode &&) = default;

  static void deleteExprNode(ExprNode *E);

protected:
  ExprNode(NodeKind K) : Kind(K) {}
  ~ExprNode() = default;
};

class CompareNode : public ExprNode {
public:
  enum ComparisonKind {

    // The comparison has a `==` operator between LHS and RHS
    Comparison_Equal,

    // The comparison has a `!=` operator between LHS and RHS
    Comparison_NotEqual,

    // The comparison is an implicit `LHS != 0`, so we can represent it with
    // just the LHS
    Comparison_NotPresent
  };

private:
  // Field that contains the compare kind
  ComparisonKind Comparison;

  // Field that contains the constant
  size_t Constant;

public:
  friend ExprNode;

  static bool classof(const ExprNode *E) {
    return E->getKind() >= NK_ValueCompare
           and E->getKind() <= NK_LoopStateCompare;
  }

protected:
  CompareNode(NodeKind Kind, ComparisonKind Comparison, size_t Constant = 0) :
    ExprNode(Kind), Comparison(Comparison), Constant(Constant) {}

  CompareNode(const CompareNode &) = default;
  CompareNode(CompareNode &&) = default;

  CompareNode() = delete;

protected:
  ~CompareNode() = default;

public:
  ComparisonKind getComparison() const { return Comparison; }

  size_t getConstant() const { return Constant; }

  void flipComparison() {
    if (Comparison == Comparison_Equal) {
      Comparison = Comparison_NotEqual;
    } else if (Comparison == Comparison_NotEqual) {
      Comparison = Comparison_Equal;
    } else if (Comparison == Comparison_NotPresent) {
      Comparison = Comparison_Equal;
      Constant = 0;
    } else {
      revng_abort();
    }
  }

  void setNotPresentKind() {
    revng_assert(Comparison != Comparison_NotPresent);
    Comparison = Comparison_NotPresent;
  }
};

class ValueCompareNode : public CompareNode {
private:
  // Pointer filed to the the llvm::BasicBlock containing the LHS of the
  // condition
  llvm::BasicBlock *BB = nullptr;

public:
  friend ExprNode;
  static bool classof(const ExprNode *E) {
    return E->getKind() == NK_ValueCompare;
  }

  ValueCompareNode(ComparisonKind Comparison,
                   llvm::BasicBlock *BB,
                   size_t Constant = 0) :
    CompareNode(NK_ValueCompare, Comparison, Constant), BB(BB) {
    revng_assert(BB != nullptr);
  }

  ValueCompareNode(const ValueCompareNode &) = default;
  ValueCompareNode(ValueCompareNode &&) = default;

  ValueCompareNode() = delete;

protected:
  ~ValueCompareNode() = default;

public:
  llvm::BasicBlock *getBasicBlock() const {
    revng_assert(BB);
    return BB;
  }
};

class LoopStateCompareNode : public CompareNode {
public:
  friend ExprNode;
  static bool classof(const ExprNode *E) {
    return E->getKind() == NK_LoopStateCompare;
  }

  LoopStateCompareNode(ComparisonKind Comparison, size_t Constant = 0) :
    CompareNode(NK_LoopStateCompare, Comparison, Constant) {}

  LoopStateCompareNode(const LoopStateCompareNode &) = default;
  LoopStateCompareNode(LoopStateCompareNode &&) = default;

  LoopStateCompareNode() = delete;

protected:
  ~LoopStateCompareNode() = default;
};

class AtomicNode : public ExprNode {
protected:
  llvm::BasicBlock *ConditionBB;

public:
  friend ExprNode;
  static bool classof(const ExprNode *E) { return E->getKind() == NK_Atomic; }

  AtomicNode(llvm::BasicBlock *BB) : ExprNode(NK_Atomic), ConditionBB(BB) {}

  AtomicNode(const AtomicNode &) = default;
  AtomicNode(AtomicNode &&) = default;

  AtomicNode() = delete;

protected:
  ~AtomicNode() = default;

public:
  llvm::BasicBlock *getConditionalBasicBlock() const { return ConditionBB; }
};

class NotNode : public ExprNode {
protected:
  ExprNode *Child;

public:
  friend ExprNode;
  static bool classof(const ExprNode *E) { return E->getKind() == NK_Not; }

  NotNode(ExprNode *N) : ExprNode(NK_Not), Child(N) {}

  NotNode(const NotNode &) = default;
  NotNode(NotNode &&) = default;

  NotNode() = delete;

protected:
  ~NotNode() = default;

public:
  ExprNode *getNegatedNode() const { return Child; }

  ExprNode **getNegatedNodeAddress() { return &Child; }

  void setNegatedNode(ExprNode *NewNegatedNode) { Child = NewNegatedNode; }
};

class BinaryNode : public ExprNode {
protected:
  ExprNode *LeftChild;
  ExprNode *RightChild;

public:
  friend ExprNode;
  static bool classof(const ExprNode *E) {
    return E->getKind() <= NK_Or and E->getKind() >= NK_And;
  }

protected:
  BinaryNode(NodeKind K, ExprNode *Left, ExprNode *Right) :
    ExprNode(K), LeftChild(Left), RightChild(Right) {}

  BinaryNode(const BinaryNode &) = default;
  BinaryNode(BinaryNode &&) = default;

  BinaryNode() = delete;
  ~BinaryNode() = default;

public:
  std::pair<ExprNode *, ExprNode *> getInternalNodes() {
    return std::make_pair(LeftChild, RightChild);
  }

  std::pair<const ExprNode *, const ExprNode *> getInternalNodes() const {
    return std::make_pair(LeftChild, RightChild);
  }

  std::pair<ExprNode **, ExprNode **> getInternalNodesAddress() {
    return std::make_pair(&LeftChild, &RightChild);
  }

  void setInternalNodes(std::pair<ExprNode *, ExprNode *> NewInternalNodes) {
    LeftChild = NewInternalNodes.first;
    RightChild = NewInternalNodes.second;
  }
};

class AndNode : public BinaryNode {
public:
  friend ExprNode;
  static bool classof(const ExprNode *E) { return E->getKind() == NK_And; }

  AndNode(ExprNode *Left, ExprNode *Right) : BinaryNode(NK_And, Left, Right) {}

  AndNode(const AndNode &) = default;
  AndNode(AndNode &&) = default;

  AndNode() = delete;

protected:
  ~AndNode() = default;
};

class OrNode : public BinaryNode {

public:
  friend ExprNode;
  static bool classof(const ExprNode *E) { return E->getKind() == NK_Or; }

  OrNode(ExprNode *Left, ExprNode *Right) : BinaryNode(NK_Or, Left, Right) {}

  OrNode(const OrNode &) = default;
  OrNode(OrNode &&) = default;

  OrNode() = delete;

protected:
  ~OrNode() = default;
};
