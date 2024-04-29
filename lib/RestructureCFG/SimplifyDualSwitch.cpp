/// \file SimplifyDualSwitch.cpp
/// Beautification pass to simplify `switch` with two entries to `if`
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

#include "revng-c/RestructureCFG/ASTNode.h"
#include "revng-c/RestructureCFG/ASTTree.h"
#include "revng-c/RestructureCFG/ExprNode.h"
#include "revng-c/Support/FunctionTags.h"

#include "SimplifyDualSwitch.h"

using namespace llvm;

static constexpr size_t SizeTMaxValue = std::numeric_limits<size_t>::max();

struct IfFields {
  ASTNode *Then = nullptr;
  ASTNode *Else = nullptr;
  size_t CaseIndex = SizeTMaxValue;
};

enum Case {
  Invalid,
  FirstCase,
  SecondCase
};

static struct std::optional<IfFields>
computeSwitchToIfPromotion(SwitchNode *Switch) {
  size_t CasesNum = Switch->cases_size();
  Case ElectedCase = Invalid;

  // We cannot promote switches with more than 2 cases
  size_t CaseNum = Switch->cases_size();
  if (CaseNum > 2)
    return std::nullopt;

  // We cannot promote non-dispatcher switches with exactly 2 cases and without
  // default, because it would be semantically incorrect, given that the
  // implicit default for a switch without a default is to jump past the switch.
  if (Switch->getCondition() != nullptr and CaseNum == 2
      and Switch->getDefault() == nullptr)
    return std::nullopt;
  revng_assert(CasesNum == 2 or CasesNum == 1);

  // If both cases are eligible, select the one with the lower label value,
  // otherwise select the only viable one, i.e., the one with a single value for
  // the label
  size_t FirstCaseSize = Switch->cases()[0].first.size();
  size_t SecondCaseSize = 0;
  if (CasesNum == 2) {
    SecondCaseSize = Switch->cases()[1].first.size();
  }
  if (FirstCaseSize == 1 and SecondCaseSize == 1) {
    revng_assert(CasesNum == 2);
    size_t FirstCaseValue = *(Switch->cases()[0].first.begin());
    size_t SecondCaseValue = *(Switch->cases()[1].first.begin());
    if (FirstCaseValue < SecondCaseValue) {
      ElectedCase = FirstCase;
    } else {
      ElectedCase = SecondCase;
    }
  } else if (FirstCaseSize == 1) {
    ElectedCase = FirstCase;
  } else if (SecondCaseSize == 1) {
    revng_assert(CasesNum == 2);
    ElectedCase = SecondCase;
  } else {
    return std::nullopt;
  }

  if (ElectedCase == Case::FirstCase) {
    return IfFields{
      .Then = Switch->cases()[0].second,
      .Else = (CasesNum == 2) ? Switch->cases()[1].second : nullptr,
      .CaseIndex = *(Switch->cases()[0].first.begin()),
    };
  } else if (ElectedCase == Case::SecondCase) {
    return IfFields{
      .Then = Switch->cases()[1].second,
      .Else = Switch->cases()[0].second,
      .CaseIndex = *(Switch->cases()[1].first.begin()),
    };
  }

  revng_abort();
}

RecursiveCoroutine<ASTNode *>
simplifyDualSwitch(ASTTree &AST, ASTNode *Node) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // In place of a sequence node, we need just to inspect all the nodes in the
    // sequence
    for (ASTNode *&N : Seq->nodes()) {
      N = rc_recur simplifyDualSwitch(AST, N);
    }

    // In this beautify, it may be that a dispatcher it is completely removed,
    // leaving a `nullptr` as an element of the containing `SequenceNode`. We
    // remove all these `nullptr`s from the `SequenceNode` after the processing
    // has taken place.
    Seq->removeNode(nullptr);
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Scs = llvm::cast<ScsNode>(Node);

    // Inspect loop nodes
    if (Scs->hasBody()) {
      ASTNode *Body = Scs->getBody();
      ASTNode *NewBody = rc_recur simplifyDualSwitch(AST, Body);
      Scs->setBody(NewBody);
    }
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // Inspect the `then` and `else` branches
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      ASTNode *NewThen = rc_recur simplifyDualSwitch(AST, Then);
      If->setThen(NewThen);
    }
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      ASTNode *NewElse = rc_recur simplifyDualSwitch(AST, Else);
      If->setElse(NewElse);
    }
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);

    // First of all, we recursively process the `case` nodes contained in the
    // `switch` in order to process the inner portion of the AST
    for (auto &LabelCasePair : Switch->cases()) {
      LabelCasePair.second = rc_recur simplifyDualSwitch(AST,
                                                         LabelCasePair.second);
    }

    // We perform the promotion of the `switch`es according to the following
    // categorization: we try to match the `switch` so that we can represent it
    // with a `if then else` logic. Then, depending on whether the `switch` is
    // A) a dispatcher or B) a standard one, we need to instantiate correctly
    // the corresponding `IfNode`, because in the dispatcher situation we do not
    // have an associated `Condition`.
    ASTNode *DefaultCase = Switch->getDefault();

    // Special case, If the only `case` present is the `default` one, we
    // should promote the body of it in place of the entire `switch`, since the
    // body of the `default` would anyway be executed
    if (Switch->cases_size() == 1 and Switch->cases()[0].first.size() == 0) {
      ASTNode *DefaultBody = Switch->cases()[0].second;
      rc_return DefaultBody;
    }

    // Try the `switch` to `if` promotion
    auto Fields = computeSwitchToIfPromotion(Switch);
    if (not Fields) {
      rc_return Switch;
    }

    using UniqueExpr = ASTTree::expr_unique_ptr;
    using ExprDestruct = ASTTree::expr_destructor;
    using ComparisonKind = CompareNode::ComparisonKind;
    ASTTree::ast_unique_ptr ASTObject;

    if (Switch->getCondition() == nullptr) {
      // A) Dispatcher `switch`.
      // Switches representing dispatchers, should not have a default case and
      // an associated `OriginalBB`.
      revng_assert(DefaultCase == nullptr);
      revng_assert(Switch->getOriginalBB() == nullptr);

      // Build the `ExprNode` containing the newly crafted `CompareNode`.
      UniqueExpr
        CondExpr(new LoopStateCompareNode(ComparisonKind::Comparison_Equal,
                                          Fields->CaseIndex),
                 ExprDestruct());
      ExprNode *Cond = AST.addCondExpr(std::move(CondExpr));
      ASTObject.reset(new IfNode(Cond, Fields->Then, Fields->Else));
    } else {
      // B) Standard `switch`.
      // Retrieve the original `BasicBlock pointed by the `switch`.
      BasicBlock *BB = Switch->getOriginalBB();
      revng_assert(BB != nullptr);
      bool IsWeaved = Switch->isWeaved();
      std::string SwitchName = "original switch name: " + Switch->getName();

      // Build the `CompareNode` equivalent to the condition of the simplified
      // switch.
      UniqueExpr CondExpr(new ValueCompareNode(ComparisonKind::Comparison_Equal,
                                               BB,
                                               Fields->CaseIndex),
                          ExprDestruct());
      ExprNode *Cond = AST.addCondExpr(std::move(CondExpr));
      ASTObject.reset(new IfNode(Cond,
                                 Fields->Then,
                                 Fields->Else,
                                 SwitchName,
                                 IsWeaved,
                                 BB));
    }

    // Assign the `if` which substitutes the `switch`
    IfNode *If = llvm::cast<IfNode>(AST.addASTNode(std::move(ASTObject)));
    revng_assert(If);

    // Remove possible `SwitchBreak` nodes that are left around in the `then` or
    // `else` branches of `if` resulting from the promotion of the `switch`.
    // The `else` branch may not be present (simplification of a `switch` with a
    // single case).
    revng_assert(Fields->Then != nullptr);

    if (auto *SwitchBreak = llvm::dyn_cast<SwitchBreakNode>(Fields->Then)) {
      revng_assert(SwitchBreak->getParentSwitch() == Switch);
      If->setThen(nullptr);
    }
    if (Fields->Else) {
      if (auto *SwitchBreak = llvm::dyn_cast<SwitchBreakNode>(Fields->Else)) {
        revng_assert(SwitchBreak->getParentSwitch() == Switch);
        If->setElse(nullptr);
      }
    }

    // After the `SwitchBreakNode` removal, it may be that the simplified
    // dispatcher `if ` becomes empty. In this case, we return `nullptr` to the
    // upper level to simplify away completely this node.
    if (not If->getThen() and not If->getElse()) {
      rc_return nullptr;
    }

    rc_return If;
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
