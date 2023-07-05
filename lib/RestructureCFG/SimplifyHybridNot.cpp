/// \file SimplifyHybridNot.cpp
/// Beautification pass to simplify hybrid `not`s
///

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/Assert.h"

#include "revng-c/RestructureCFG/ASTNode.h"
#include "revng-c/RestructureCFG/ASTTree.h"
#include "revng-c/RestructureCFG/ExprNode.h"
#include "revng-c/RestructureCFG/SimplifyHybridNot.h"
#include "revng-c/Support/FunctionTags.h"
using namespace llvm;

struct AssociatedExprs {
  llvm::SmallVector<ExprNode **> DirectExprs;
  llvm::SmallVector<ExprNode **> NegatedExprs;
};

using BBExprsMap = llvm::SmallDenseMap<BasicBlock *, AssociatedExprs>;

enum Direction {
  Direct,
  Negated
};

static void insertInAssociatedExprs(BBExprsMap &BBExprs,
                                    BasicBlock *BB,
                                    ExprNode **Node,
                                    Direction Direction) {

  // Check if the entry in the map is already present
  BBExprsMap::iterator It = BBExprs.find(BB);
  if (It == BBExprs.end()) {

    // Insert in the map the corresponding entry if not already present, and
    // reassing the iterator so we can use it
    It = BBExprs.insert(std::make_pair(BB, AssociatedExprs())).first;
  }

  // Insert the pointer to the `ExprNode` in the correct vector
  if (Direction == Direct) {
    It->second.DirectExprs.push_back(Node);
  } else if (Direction == Negated) {
    It->second.NegatedExprs.push_back(Node);
  } else {
    revng_abort();
  }
}

static RecursiveCoroutine<void> collectExprBB(ExprNode **Expr,
                                              BBExprsMap &BBExprs) {

  switch ((*Expr)->getKind()) {
  case ExprNode::NodeKind::NK_Atomic: {
    auto *Atomic = llvm::cast<AtomicNode>(*Expr);
    BasicBlock *BB = Atomic->getConditionalBasicBlock();

    // Insert the `ExprNode`
    insertInAssociatedExprs(BBExprs, BB, Expr, Direct);
  } break;

  case ExprNode::NodeKind::NK_Not: {
    auto *Not = llvm::cast<NotNode>(*Expr);

    if (auto *Contained = llvm::dyn_cast<AtomicNode>(Not->getNegatedNode())) {
      BasicBlock *BB = Contained->getConditionalBasicBlock();

      // The `NotNode` directly contains an `AtomicNode`, we therefore insert
      // the address to the `NotNode` in the `NegatedExprs` vector
      insertInAssociatedExprs(BBExprs, BB, Expr, Negated);
    } else {

      // If the `NotNode` does not directly contain an `AtomicNode`, we need to
      // continue with the inspection
      ExprNode **NegatedNode = Not->getNegatedNodeAddress();
      rc_recur collectExprBB(NegatedNode, BBExprs);
    }
  } break;

  case ExprNode::NodeKind::NK_And:
  case ExprNode::NodeKind::NK_Or: {
    auto *Binary = llvm::cast<BinaryNode>(*Expr);
    const auto &[LHS, RHS] = Binary->getInternalNodesAddress();
    rc_recur collectExprBB(LHS, BBExprs);
    rc_recur collectExprBB(RHS, BBExprs);
  } break;

  default:
    revng_abort();
  }
  rc_return;
}

static RecursiveCoroutine<void>
populateAssociatedExprMap(ASTTree &AST, ASTNode *Node, BBExprsMap &BBExprs) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // Recursively call the visit on each element of the `SequenceNode`
    for (ASTNode *&N : Seq->nodes()) {
      rc_recur populateAssociatedExprMap(AST, N, BBExprs);
    }
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Scs = llvm::cast<ScsNode>(Node);

    // In case we are inspecting a loop that has been promoted to `while` or
    // `dowhile`, we should inspect the related condition containing the
    // `IfNode` associated to the execution of the loop
    if (not Scs->isWhileTrue()) {
      IfNode *If = Scs->getRelatedCondition();
      ExprNode **IfExpr = If->getCondExprAddress();
      collectExprBB(IfExpr, BBExprs);
    }

    if (Scs->hasBody()) {
      rc_recur populateAssociatedExprMap(AST, Scs->getBody(), BBExprs);
    }
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);
    ExprNode **IfExpr = If->getCondExprAddress();
    collectExprBB(IfExpr, BBExprs);

    if (If->hasThen()) {
      rc_recur populateAssociatedExprMap(AST, If->getThen(), BBExprs);
    }
    if (If->hasElse()) {
      rc_recur populateAssociatedExprMap(AST, If->getElse(), BBExprs);
    }
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);
    for (auto &LabelCasePair : Switch->cases()) {
      ASTNode *Case = LabelCasePair.second;
      rc_recur populateAssociatedExprMap(AST, Case, BBExprs);
    }
  } break;
  case ASTNode::NK_Code:
  case ASTNode::NK_Set:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break:

    // These nodes do not have an `ExprNode` embedded, nor do embed other
    // nested nodes
    break;
  default:
    revng_unreachable();
  }

  rc_return;
}

enum NotKind {
  SimpleIR,
  BooleanNot
};

using BBSet = llvm::SmallPtrSet<BasicBlock *, 4>;

using ConsensusMap = llvm::SmallDenseMap<BasicBlock *, NotKind>;

static ConsensusMap computeBBConsensus(BBExprsMap &BBExprs) {
  ConsensusMap ConsensusBB;

  // We inspect both the IR and the associated `ExprNode`s.
  // Specifically, we need to check the following:
  // 1) The `ICmpInst` present on the IR is either and equality or a
  //    disequality.
  // 2) The number of `Direct` and `Negated` `Expr`s is coherent with the
  //    flip operation that we want to conduct.
  for (const auto &[BB, AssociatedExprs] : BBExprs) {
    Instruction *Terminator = BB->getTerminator();
    BranchInst *Branch = llvm::cast<BranchInst>(Terminator);

    // We expect that the BasicBlock associated to an `IfNode` `ExprNode`, does
    // contain a conditional branch
    revng_assert(Branch->isConditional());

    // Obtain the comparison instruction and compute the direction of the
    // transformation
    llvm::Value *Condition = Branch->getCondition();

    auto &DirectExprs = AssociatedExprs.DirectExprs;
    auto &NegatedExprs = AssociatedExprs.NegatedExprs;

    // TODO: we currently handle `ICmpInst`s and `BooleanNot`s as conditions for
    //       the branch, explore alternative situations that we may want to
    //       handle. Doing that would however mean performing hybrid analysis on
    //       both the IR level and `ExprNode`s level, therefore increasing
    //       significantly the complexity space of the analysis. We leave this
    //       as a future enhancement.
    if (llvm::isa<ICmpInst>(Condition)) {

      auto *Compare = llvm::cast<ICmpInst>(Condition);

      // Analyze the comparison predicate, to distinguish the following
      // situations:
      // 1) The IR comparison is a `NE`. In this situation, if the associated
      // `Negated` expressions is > than the number of the associated `Direct`
      // expressions, we can flip IR and `ExprNode` conditions. In case of a
      // tie, we perform the transformation in order to minimize the number of
      // adjacent negations. 2) The IR comparison is a `EQ`. In this situation,
      // if the associated `Negated` expressions are > than the number of the
      // associated `Direct` expressions, we can flip IR and `ExprNode`
      // conditions. In case of a tie, we do not perform the transformation, in
      // order to minimize the number of adjacent negations.
      auto Predicate = Compare->getPredicate();

      if (Predicate == llvm::ICmpInst::Predicate::ICMP_NE) {
        if (NegatedExprs.size() >= DirectExprs.size()) {
          ConsensusBB.insert(std::make_pair(BB, NotKind::SimpleIR));
        }
      }
      if (Predicate == llvm::ICmpInst::Predicate::ICMP_EQ) {
        if (NegatedExprs.size() > DirectExprs.size()) {
          ConsensusBB.insert(std::make_pair(BB, NotKind::SimpleIR));
        }
      }
    } else if (isCallToTagged(Condition, FunctionTags::BooleanNot)) {

      // Handle the hybrid simplify starting the analysis from the `BooleanNot`
      // call in the LLVM IR
      if (NegatedExprs.size() >= DirectExprs.size()) {
        ConsensusBB.insert(std::make_pair(BB, NotKind::BooleanNot));
      }
    }
  }
  return ConsensusBB;
}

static llvm::Value *getCondition(BasicBlock *BB) {
  Instruction *Terminator = BB->getTerminator();
  BranchInst *Branch = llvm::cast<BranchInst>(Terminator);

  // We should ensure that the current `BranchInst` is indeed a conditional
  // branch
  revng_assert(Branch->isConditional());

  return Branch->getCondition();
}

static void invertPredicate(llvm::Value *Condition) {

  // Perform the inversion of the comparison predicate in the related
  // BasicBlock, i.e., not equality is transformed into equality, and
  // vice-versa
  auto *Compare = llvm::cast<ICmpInst>(Condition);
  Compare->setPredicate(Compare->getInversePredicate());
}

static void flipIRNot(BasicBlock *BB, const NotKind &NotKind) {
  if (NotKind == NotKind::SimpleIR) {

    // Go back up in order to find the comparison instruction and check that is
    // suitable for flipping
    llvm::Value *Condition = getCondition(BB);
    invertPredicate(Condition);
  } else if (NotKind == NotKind::BooleanNot) {

    // In this situation, we inspect the IR in order to find the original `icmp`
    // instruction referenced by the `BooleanNot` opcode, we perform the flip of
    // that comparison, and remove the `BooleanNot` opcode from the IR
    llvm::Value *Condition = getCondition(BB);
    llvm::CallInst *Call = getCallToTagged(Condition, FunctionTags::BooleanNot);
    revng_assert(Call);

    // We manually forge the new `icmp ne 0` to represent the inversion of the
    // `@boolean_not` predicate semantics
    llvm::Value *OriginalLHS = Call->getArgOperand(0);
    llvm::IRBuilder<> Builder(BB);
    Builder.SetInsertPoint(Call);
    llvm::Value *NewCondition = Builder.CreateIsNotNull(OriginalLHS);

    // Substitute in the `BranchInst` the condition with the inverted comparison
    // predicate
    cast<BranchInst>(BB->getTerminator())->setCondition(NewCondition);

    // Remove the `BooleanNot` only if there is no other live use apart from the
    // one used in the condition of the branch
    llvm::RecursivelyDeleteTriviallyDeadInstructions(Call);
  } else {
    revng_abort();
  }

  return;
}

static void
flipAssociatedExprs(ASTTree &AST, BBExprsMap &BBExprs, BasicBlock *BB) {
  // Flip the condition on the expressions associated to the BasicBlock under
  // analysis
  auto It = BBExprs.find(BB);
  revng_assert(It != BBExprs.end());

  auto &AssociatedExprs = It->second;
  auto &DirectExprs = AssociatedExprs.DirectExprs;
  auto &NegatedExprs = AssociatedExprs.NegatedExprs;

  // Iterate over the sets of the direct and negated associated expressions
  for (ExprNode **DirectExpr : DirectExprs) {
    using UniqueExpr = ASTTree::expr_unique_ptr;
    UniqueExpr Not;
    Not.reset(new NotNode(*DirectExpr));
    ExprNode *NotNode = AST.addCondExpr(std::move(Not));
    *DirectExpr = NotNode;
  }

  for (ExprNode **NegatedExpr : NegatedExprs) {
    NotNode *ContainedNot = llvm::cast<NotNode>(*NegatedExpr);
    ExprNode *AtomicNode = ContainedNot->getNegatedNode();
    *NegatedExpr = AtomicNode;
  }
}

static void simplifyHybridNotImpl(ASTTree &AST,
                                  BBExprsMap &BBExprs,
                                  ConsensusMap &ConsensusBB) {
  for (const auto &[BB, NotKind] : ConsensusBB) {

    // Flip the condition on the LLVMIR
    flipIRNot(BB, NotKind);

    // Flip the condition on the `ExprNode`s
    flipAssociatedExprs(AST, BBExprs, BB);
  }

  return;
}

ASTNode *simplifyHybridNot(ASTTree &AST, ASTNode *RootNode) {
  // The role of this function is to perform the double `not` simplification.
  // Our goal is to collect the negation both on the GHAST level (the `NotNode`
  // contained in the `ExprNode` associated to the condition we want to explore)
  // and the negation that is represented on the IR level with a comparison
  // which implies a negation.
  BBExprsMap BBExprs;

  // Map that contains the correspondence of the `ExprNode`s affected by a
  // BasicBlock
  populateAssociatedExprMap(AST, RootNode, BBExprs);

  // Run the analysis which checks if all the references to a single instance of
  // block that is a candidate for flipping, do agree for the flip operation
  ConsensusMap ConsensusBB = computeBBConsensus(BBExprs);

  // Perform the simplification for the BBs for which the consensus computation
  // agrees on the outcome of the transformation
  simplifyHybridNotImpl(AST, BBExprs, ConsensusBB);

  return RootNode;
}
