/// \file SimplifyImplicitStatement.cpp
/// Beautification pass to make superfluous statement implicit, either a
/// `continue` or a `return` in the current implementation
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

#include "SimplifyImplicitStatement.h"

using namespace llvm;

enum StatementType {
  ST_Continue,
  ST_Return
};

template<StatementType Ty>
static RecursiveCoroutine<bool>
simplifyImplicitStatementImpl(ASTTree &AST,
                              ASTNode *Node,
                              bool SuccessorEmpty) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // We need to iterate over the `SequenceNode` in reverse order, passing the
    // result each time to the next(previous) node
    bool SequenceEmpty = SuccessorEmpty;
    for (ASTNode *N : llvm::reverse(Seq->nodes())) {
      SequenceEmpty = SequenceEmpty
                      and rc_recur
                        simplifyImplicitStatementImpl<Ty>(AST,
                                                          N,
                                                          SequenceEmpty);

      // Optimization to avoid entire traversal of the `ASTTree`. Once the
      // `SuccessorEmpty` parameter assumes the `false` state, we cannot go back
      // to `true`, because we have found a point in the `sequence` where we do
      // not have an empty successor. We can therefore break out from the
      // function.
      if (SequenceEmpty == false) {
        break;
      }
    }

    rc_return SequenceEmpty;
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Scs = llvm::cast<ScsNode>(Node);

    // Inspect loop nodes
    if (Scs->hasBody()) {
      ASTNode *Body = Scs->getBody();
      rc_return rc_recur simplifyImplicitStatementImpl<Ty>(AST,
                                                           Body,
                                                           SuccessorEmpty);
    } else {
      rc_return false;
    }
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // Process before the `else`, if present, and later the `then`, and return
    // the logical `and` of the two
    bool ElseEmpty = true;
    bool ThenEmpty = true;
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      ElseEmpty = rc_recur simplifyImplicitStatementImpl<Ty>(AST,
                                                             Else,
                                                             SuccessorEmpty);
    }

    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      ThenEmpty = rc_recur simplifyImplicitStatementImpl<Ty>(AST,
                                                             Then,
                                                             SuccessorEmpty);
    }

    rc_return ElseEmpty and ThenEmpty;
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);

    // Process the `switch` cases, and return the logical `and` of all the
    // values assumed by them
    bool CasesEmpty = true;
    for (auto &LabelCasePair : Switch->cases()) {
      CasesEmpty = CasesEmpty
                   and rc_recur
                     simplifyImplicitStatementImpl<Ty>(AST,
                                                       LabelCasePair.second,
                                                       SuccessorEmpty);
    }

    rc_return CasesEmpty;
  } break;
  case ASTNode::NK_Continue: {
    if constexpr (Ty == StatementType::ST_Continue) {
      ContinueNode *Continue = cast<ContinueNode>(Node);

      // If the successor is empty, it means that we can promote the current
      // `continue` to an implicit one
      if (SuccessorEmpty == true) {
        Continue->setImplicit();
        rc_return true;
      }
      rc_return false;
    } else if constexpr (Ty == StatementType::ST_Return) {
      rc_return false;
    } else {
      revng_abort();
    }
  } break;
  case ASTNode::NK_Code: {
    if constexpr (Ty == StatementType::ST_Continue) {
      rc_return false;
    } else if constexpr (Ty == StatementType::ST_Return) {
      auto *Code = llvm::cast<CodeNode>(Node);

      // Inspect the related IR block to search for a `VoidTy`
      // `llvm::ReturnInst`
      BasicBlock *OriginalBB = Code->getOriginalBB();
      revng_assert(OriginalBB != nullptr);

      Instruction &LastInstruction = OriginalBB->back();

      // The current `CodeNode` is candidate for the implicit `return`
      // simplification only if there `LastCodeNode` parameter is still null,
      // which means that no previous `CodeNode` has been inspected yet
      if (SuccessorEmpty == true) {
        if (auto *Return = dyn_cast<ReturnInst>(&LastInstruction)) {

          // Verify that the `llvm::ReturnInst` is a `VoidTy` `return`, which is
          // the only `return` type that can be simplified
          if (Return->getReturnValue() == nullptr) {
            Code->setImplicitReturn();
            rc_return true;
          }
        }
      }

      rc_return false;
    } else {
      revng_abort();
    }

  } break;
  case ASTNode::NK_Set:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Break: {
    // All these emit a statement
    rc_return false;
  } break;
  default:
    revng_unreachable();
  }

  revng_abort();
}

void simplifyImplicitContinue(ASTTree &AST) {
  for (ASTNode *Node : AST.nodes()) {
    auto *Scs = dyn_cast<ScsNode>(Node);
    if (not Scs or not Scs->hasBody())
      continue;

    // Explore the body of the `Scs` to find nodes that can be
    // simplified
    ASTNode *Body = Scs->getBody();
    simplifyImplicitStatementImpl<StatementType::ST_Continue>(AST, Body, true);
  }
}

void simplifyImplicitReturn(ASTTree &AST, ASTNode *Node) {
  // Start the recursive exploration of the `ASTTree`, signaling that currently
  // the "next" node is non existent.
  simplifyImplicitStatementImpl<StatementType::ST_Return>(AST, Node, true);
}
