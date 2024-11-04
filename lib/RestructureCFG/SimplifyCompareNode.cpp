/// \file SimplifyCompareNode.cpp
/// Beautification pass to simplify `CompareNode`
///

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/RestructureCFG/ASTNode.h"
#include "revng-c/RestructureCFG/ASTTree.h"
#include "revng-c/RestructureCFG/ExprNode.h"

#include "SimplifyCompareNode.h"

using namespace llvm;

RecursiveCoroutine<ASTNode *> simplifyCompareNode(ASTTree &AST, ASTNode *Node) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // In place of a sequence node, we need just to inspect all the nodes in the
    // sequence
    for (ASTNode *&N : Seq->nodes()) {
      N = rc_recur simplifyCompareNode(AST, N);
    }
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Scs = llvm::cast<ScsNode>(Node);

    // Inspect loop nodes
    if (Scs->hasBody()) {
      ASTNode *Body = Scs->getBody();
      ASTNode *NewBody = rc_recur simplifyCompareNode(AST, Body);
      Scs->setBody(NewBody);
    }
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // Inspect the `then` and `else` branches
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      ASTNode *NewThen = rc_recur simplifyCompareNode(AST, Then);
      If->setThen(NewThen);
    }
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      ASTNode *NewElse = rc_recur simplifyCompareNode(AST, Else);
      If->setElse(NewElse);
    }

    // If the associated `CondExpr` contains a `NotNode`, which in turn contains
    // a `CompareNode` containing an `Equal` expression, we can remove the
    // `NotNode` altogether and transform the `CompareNode` into a `not equal`.
    // Same applies for a `NotNode` containing a `NotEqual` `CompareNode`.
    ExprNode *IfCondExpr = If->getCondExpr();
    if (auto *Not = llvm::dyn_cast<NotNode>(IfCondExpr)) {
      ExprNode *NegatedExpr = Not->getNegatedNode();
      revng_assert(NegatedExpr);
      if (auto *Compare = llvm::dyn_cast<CompareNode>(NegatedExpr)) {
        Compare->flipComparison();
        If->replaceCondExpr(Compare);
      }
    }

    // Further simplification for special `CompareNode`s comparing with constant
    // `0`. Specifically:
    // - If the associated CondExpr` contains a `CompareNode`, which is `LHS ==
    //   0`, we convert it to a `NotNode` containing a `CompareNode` of the
    //   `NotPresent` kind
    // - Equally, a `CompareNode`, which is `LHS != 0`, we convert it to a
    //   `CompareNode` of the `NotPresent` kind.
    IfCondExpr = If->getCondExpr();
    if (auto *Compare = llvm::dyn_cast<CompareNode>(IfCondExpr)) {
      if (Compare->getConstant() == 0) {
        using ComparisonKind = CompareNode::ComparisonKind;
        auto Comparison = Compare->getComparison();
        if (Comparison == ComparisonKind::Comparison_Equal) {
          Compare->setNotPresentKind();
          using UniqueExpr = ASTTree::expr_unique_ptr;
          UniqueExpr Not;
          Not.reset(new NotNode(Compare));
          ExprNode *NotNode = AST.addCondExpr(std::move(Not));
          If->replaceCondExpr(NotNode);
        } else if (Comparison == ComparisonKind::Comparison_NotEqual) {
          Compare->setNotPresentKind();
        }
      }
    }

  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);

    // First of all, we recursively process the `case` nodes contained in the
    // `switch` in order to process the inner portion of the AST
    for (auto &LabelCasePair : Switch->cases()) {
      LabelCasePair.second = rc_recur simplifyCompareNode(AST,
                                                          LabelCasePair.second);
    }
  } break;
  case ASTNode::NK_Code:
  case ASTNode::NK_Set:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break:
    // Do nothing
    break;
  default:
    revng_unreachable();
  }

  rc_return Node;
}
