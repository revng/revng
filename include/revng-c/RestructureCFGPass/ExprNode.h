#ifndef REVNGC_RESTRUCTURE_CFG_EXPRNODE_H
#define REVNGC_RESTRUCTURE_CFG_EXPRNODE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdlib>

// LLVM includes
#include <llvm/Support/Casting.h>

class ExprNode {
public:
  enum NodeKind { NK_Atomic, NK_Not, NK_And, NK_Or };

private:
  const NodeKind Kind;

public:
  NodeKind getKind() const { return Kind; }

  virtual ~ExprNode() = default;

protected:
  ExprNode(NodeKind K) : Kind(K) {}
};

class AtomicNode : public ExprNode {
private:
  llvm::BasicBlock *ConditionBB;

public:
  AtomicNode(llvm::BasicBlock *BB) : ExprNode(NK_Atomic), ConditionBB(BB) {}

  virtual ~AtomicNode() override = default;

  static bool classof(const ExprNode *E) { return E->getKind() == NK_Atomic; }

  llvm::BasicBlock *getConditionalBasicBlock() const { return ConditionBB; }
};

class NotNode : public ExprNode {
private:
  ExprNode *Child;

public:
  NotNode(ExprNode *N) : ExprNode(NK_Not), Child(N) {}

  virtual ~NotNode() override = default;

  static bool classof(const ExprNode *E) { return E->getKind() == NK_Not; }

  ExprNode *getNegatedNode() const { return Child; }
};

class BinaryNode : public ExprNode {
private:
  ExprNode *LeftChild;
  ExprNode *RightChild;

public:
  std::pair<ExprNode *, ExprNode *> getInternalNodes() {
    return std::make_pair(LeftChild, RightChild);
  }

  static bool classof(const ExprNode *E) {
    return E->getKind() <= NK_Or and E->getKind() >= NK_And;
  }

  virtual ~BinaryNode() override = default;

protected:
  BinaryNode(NodeKind K, ExprNode *Left, ExprNode *Right) :
    ExprNode(K),
    LeftChild(Left),
    RightChild(Right) {}
};

class AndNode : public BinaryNode {

public:
  AndNode(ExprNode *Left, ExprNode *Right) : BinaryNode(NK_And, Left, Right) {}

  virtual ~AndNode() override = default;

  static bool classof(const ExprNode *E) { return E->getKind() == NK_And; }
};

class OrNode : public BinaryNode {

public:
  OrNode(ExprNode *Left, ExprNode *Right) : BinaryNode(NK_Or, Left, Right) {}

  virtual ~OrNode() override = default;

  static bool classof(const ExprNode *E) { return E->getKind() == NK_Or; }
};

#endif // define REVNGC_RESTRUCTURE_CFG_EXPRNODE_H
