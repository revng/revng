/// \file RemoveDeadCode.cpp
/// Beautification pass to simplify dead code in the AST representation
///

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/Assert.h"

#include "revng-c/RestructureCFG/ASTNode.h"
#include "revng-c/RestructureCFG/ASTTree.h"

#include "FallThroughScopeAnalysis.h"
#include "RemoveDeadCode.h"

static RecursiveCoroutine<ASTNode *>
removeDeadCodeImpl(ASTNode *Node,
                   const FallThroughScopeTypeMap &FallThroughScopeMap) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // In place of a sequence node, we just need to inspect all the nodes in the
    // sequence
    bool ShouldSimplify = false;
    for (ASTNode *&N : Seq->nodes()) {
      if (not ShouldSimplify) {

        // Compute the `FallThrough information after each `SequencenNode` is
        // processed, and mark the code alive from that node in the sequence on
        // only if we have fallthrough
        N = rc_recur removeDeadCodeImpl(N, FallThroughScopeMap);
        FallThroughScopeType
          FallThroughType = FallThroughScopeType::FallThrough;
        ShouldSimplify = FallThroughScopeMap.at(N) != FallThroughType;
      } else {
        N = nullptr;
      }
    }

    // In this beautify, it may be that a dispatcher switch is completely
    // removed, therefore leaving a `nullptr` in a `SequenceNode`. We remove all
    // these `nullptr` from the `SequenceNode` after the processing has taken
    // place
    Seq->removeNode(nullptr);

    // This simplify should never leave an empty `SequenceNode` (at the very
    // least, the first nody of the `SequenceNode`, which is the one not
    // performing fallthrough, should be kept alive). Later on, in the beautify
    // pass "pipeline", we have an invocation of the `simplifyAtomicSequence`
    // routine aimed to unwrap unitary `SequenceNode`s.
    revng_assert(not Seq->empty());
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Scs = llvm::cast<ScsNode>(Node);

    // Inspect loop nodes
    if (Scs->hasBody()) {
      ASTNode *Body = Scs->getBody();
      ASTNode *NewBody = rc_recur removeDeadCodeImpl(Body, FallThroughScopeMap);
      Scs->setBody(NewBody);
    }

  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // Inspect the `then` and `else` branches
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      ASTNode *NewThen = rc_recur removeDeadCodeImpl(Then, FallThroughScopeMap);
      If->setThen(NewThen);
    }
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      ASTNode *NewElse = rc_recur removeDeadCodeImpl(Else, FallThroughScopeMap);
      If->setElse(NewElse);
    }
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);

    // First of all, we recursively process the `case` nodes contained in the
    // `switch` in order to process the inner portion of the AST
    llvm::SmallVector<size_t> ToRemoveCaseIndex;
    for (auto &Group : llvm::enumerate(Switch->cases())) {
      unsigned Index = Group.index();
      auto &LabelCasePair = Group.value();
      LabelCasePair.second = rc_recur removeDeadCodeImpl(LabelCasePair.second,
                                                         FallThroughScopeMap);

      if (LabelCasePair.second == nullptr) {
        ToRemoveCaseIndex.push_back(Index);
      }
    }

    for (auto ToRemoveCase : llvm::reverse(ToRemoveCaseIndex)) {
      Switch->removeCaseN(ToRemoveCase);
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

/// This simplification routine mimics a dead code elimination pass. Basically,
/// when we have a `SequenceNode`, and we find a node sporting the
/// `nofallthrough` beahavior, we can remove all the following nodes in the
/// `SequenceNode`.
/// A practical example of why this may happen: after the inlining of dispatcher
/// `case`s, if a `return` statement is moved into an inner loop in place of a
/// `SetNode`, it may be that a following `break` statement, and therefore the
/// `break` can be simplified away.
ASTNode *removeDeadCode(const model::Binary &Model, ASTTree &AST) {
  ASTNode *RootNode = AST.getRoot();

  // Pre-compute the `FallThroughScopeType` before the `SuperfluousNonLocalCF`
  // simplify is performed. This information is useful in order to decide
  // whether we remove some statements that after the `case` inlining are
  // preceded by `return` statements.
  FallThroughScopeTypeMap
    FallThroughScopeMap = computeFallThroughScope(Model, RootNode);

  // Perform the `SuperfluousNonLocalCF` simplification pass
  RootNode = removeDeadCodeImpl(RootNode, FallThroughScopeMap);

  // Update the root field of the AST
  AST.setRoot(RootNode);

  return RootNode;
}
