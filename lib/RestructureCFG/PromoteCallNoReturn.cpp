/// \file PromoteCallNoReturn.cpp
/// Beautification pass to perform the `CallNoReturn` promotion
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
#include "revng-c/RestructureCFG/ASTNodeUtils.h"
#include "revng-c/RestructureCFG/ASTTree.h"
#include "revng-c/RestructureCFG/ExprNode.h"
#include "revng-c/RestructureCFG/GenerateAst.h"
#include "revng-c/Support/FunctionTags.h"

#include "FallThroughScopeAnalysis.h"
#include "PromoteCallNoReturn.h"

using namespace llvm;

static bool isPreferredAsFallThrough(FallThroughScopeType Element) {
  switch (Element) {
  case FallThroughScopeType::Return:
  case FallThroughScopeType::Continue:
  case FallThroughScopeType::LoopBreak: {
    return true;
  } break;
  case FallThroughScopeType::FallThrough:
  case FallThroughScopeType::MixedNoFallThrough:
  case FallThroughScopeType::CallNoReturn: {
    return false;
  } break;
  }
}

static RecursiveCoroutine<ASTNode *>
promoteCallNoReturnImpl(ASTTree &AST,
                        ASTNode *Node,
                        FallThroughScopeTypeMap &FallThroughScopeMap) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // In place of a sequence node, we need just to inspect all the nodes in the
    // sequence
    for (ASTNode *&N : Seq->nodes()) {
      N = rc_recur promoteCallNoReturnImpl(AST, N, FallThroughScopeMap);
    }

    // The general idea of the implementation is the following:
    // when the `SequenceNode` ends with a `CallNoReturn` scope, we search for a
    // preceding `IfNode` with just the `then` scope, that ends if a
    // `NonLocalCF` scope. If we can find such situation, we swap the
    // statements, in order to try to _push down_ the `NonLocalCF` statements.
    // At this stage, we assume that every possible `IfNode` we encounter, has
    // already been normalized with respect the empty `then` criterion.
    size_t SequenceSize = Seq->length();

    // It would be useless to run the analysis on a sequence with a single
    // element
    if (SequenceSize > 1) {
      ASTNode *LastSequenceNode = Seq->getNodeN(SequenceSize - 1);
      FallThroughScopeType LastSequenceNodeScopeType = FallThroughScopeMap
                                                         .at(LastSequenceNode);
      if (LastSequenceNodeScopeType == FallThroughScopeType::CallNoReturn) {

        // We go back and search for the `IfNode`
        for (ptrdiff_t Index = static_cast<ptrdiff_t>(SequenceSize - 2);
             Index >= 0;
             Index--) {
          ASTNode *PreviousNode = Seq->getNodeN(Index);
          if (auto *If = llvm::dyn_cast<IfNode>(PreviousNode)) {
            ASTNode *IfThen = If->getThen();
            if (not If->hasElse()
                and isPreferredAsFallThrough(FallThroughScopeMap.at(IfThen))) {

              // We matched the conditions, so we proceed with the
              // transformation. We insert the body of the `then` in the back of
              // the `SequenceNode`.
              Seq->addNode(IfThen);
              If->setThen(nullptr);

              // We iterate over all the nodes in the `SequenceNode` that
              // follows the `IfNode` (excluding the newly added one). These
              // nodes, will constitute the body of the `else` of the `IfNode`.
              // A later invocation of the `flipEmptyTyen` pass, will be
              // responsible for normalizing this in the expected form.
              llvm::SmallVector<ASTNode *> ToRemove;
              SequenceNode *IfThenSequence = AST.addSequenceNode();
              for (size_t FollowingNodeIndex = Index + 1;
                   FollowingNodeIndex < SequenceSize;
                   FollowingNodeIndex++) {
                ASTNode *Elem = Seq->getNodeN(FollowingNodeIndex);
                IfThenSequence->addNode(Elem);
                ToRemove.push_back(Elem);
              }

              // Remove the nodes from the original `SequenceNode`
              for (auto *Elem : llvm::reverse(ToRemove)) {
                Seq->removeNode(Elem);
              }

              If->setElse(IfThenSequence);

              // If we performed the promotion, we should not proceed with the
              // iteration over the `SequenceNode`. First, beceause the iterator
              // would be invalid now, second, because we would not find any
              // opportunity for another promotion.
              break;
            }
          }
        }
      }
    }
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Scs = llvm::cast<ScsNode>(Node);

    // Inspect loop nodes
    if (Scs->hasBody()) {
      ASTNode *Body = Scs->getBody();
      ASTNode *NewBody = rc_recur promoteCallNoReturnImpl(AST,
                                                          Body,
                                                          FallThroughScopeMap);
      Scs->setBody(NewBody);
    }
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // Inspect the `then` and `else` branches
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      ASTNode *NewThen = rc_recur promoteCallNoReturnImpl(AST,
                                                          Then,
                                                          FallThroughScopeMap);
      If->setThen(NewThen);
    }
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      ASTNode *NewElse = rc_recur promoteCallNoReturnImpl(AST,
                                                          Else,
                                                          FallThroughScopeMap);
      If->setElse(NewElse);
    }
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);

    // First of all, we recursively process the `case` nodes contained in the
    // `switch` in order to process the inner portion of the AST
    for (auto &LabelCasePair : Switch->cases()) {
      LabelCasePair.second = rc_recur
        promoteCallNoReturnImpl(AST, LabelCasePair.second, FallThroughScopeMap);
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

ASTNode *promoteCallNoReturn(const model::Binary &Model,
                             ASTTree &AST,
                             ASTNode *RootNode) {

  // Perform the computation of fallthrough scopes type
  FallThroughScopeTypeMap
    FallThroughScopeMap = computeFallThroughScope(Model, RootNode);

  // Run the `PromoteCallNoReturn` transformation
  RootNode = promoteCallNoReturnImpl(AST, RootNode, FallThroughScopeMap);

  // Run the canonicalization steps. In detail, we need to run the following for
  // these reasons:
  // - The `PromoteCallNoReturn` pass, when applied, leaves a `IfNode` with the
  //   `then` body empty, so we need to normalize it.
  // - If the promotion happens, it may be that the body of the `then` of the if
  //   is appended to the sequence node containing it. If the `then` of the `if`
  //   is already a `SequenceNode` itself, means that we would have a
  //   `SequenceNode` as a direct child of another `SequenceNode`, which is
  //   situation that violates our canonicalization assumptions, since we always
  //   want to flatten directly nested `SequenceNode`s.
  // - If the promotion above happens, we need to move all the nodes in the
  //   `SequenceNode` that follow the `IfNode`. To do that, we first create a
  //   `SequenceNode` in the `else`, and add all the nodes. If it happens that
  //   there is a single node, we end up with a `SequenceNode` containing a
  //   single node, a situation that violates our canonicalization assumptions.
  RootNode = canonicalize(AST, RootNode);

  return RootNode;
}
