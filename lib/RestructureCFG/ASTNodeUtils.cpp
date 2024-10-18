/// \file ASTNodeUtils.cpp
/// Utils for AST nodes
///

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"

#include "revng-c/RestructureCFG/ASTNode.h"
#include "revng-c/RestructureCFG/ASTNodeUtils.h"
#include "revng-c/RestructureCFG/ASTTree.h"

/// Visit the node and all its children recursively, checking if a loop
/// variable is needed.
static RecursiveCoroutine<bool> needsLoopVarImpl(const ASTNode *N) {
  if (N == nullptr)
    rc_return false;

  auto Kind = N->getKind();
  switch (Kind) {

  case ASTNode::NodeKind::NK_Break:
  case ASTNode::NodeKind::NK_SwitchBreak:
  case ASTNode::NodeKind::NK_Continue:
  case ASTNode::NodeKind::NK_Code:
    rc_return false;

  case ASTNode::NodeKind::NK_If: {
    const IfNode *If = llvm::cast<IfNode>(N);

    if (nullptr != If->getThen())
      if (rc_recur needsLoopVarImpl(If->getThen()))
        rc_return true;

    if (If->hasElse())
      if (rc_recur needsLoopVarImpl(If->getElse()))
        rc_return true;

    rc_return false;
  }

  case ASTNode::NodeKind::NK_Scs: {
    const ScsNode *LoopBody = llvm::cast<ScsNode>(N);
    rc_return rc_recur needsLoopVarImpl(LoopBody->getBody());
  }

  case ASTNode::NodeKind::NK_List: {
    const SequenceNode *Seq = llvm::cast<SequenceNode>(N);
    for (ASTNode *Child : Seq->nodes())
      if (rc_recur needsLoopVarImpl(Child))
        rc_return true;

    rc_return false;
  }

  case ASTNode::NodeKind::NK_Switch: {
    const SwitchNode *Switch = llvm::cast<SwitchNode>(N);
    llvm::Value *SwitchVar = Switch->getCondition();

    if (not SwitchVar)
      rc_return true;

    for (const auto &[Labels, CaseNode] : Switch->cases_const_range())
      if (rc_recur needsLoopVarImpl(CaseNode))
        rc_return true;

    rc_return false;
  }

  case ASTNode::NodeKind::NK_Set: {
    rc_return true;
  }
  }
}

bool needsLoopVar(const ASTNode *N) {
  return needsLoopVarImpl(N);
}

using UniqueExpr = ASTTree::expr_unique_ptr;

static RecursiveCoroutine<void> flipEmptyThenImpl(ASTTree &AST, ASTNode *Node) {
  if (auto *Sequence = llvm::dyn_cast<SequenceNode>(Node)) {
    for (ASTNode *Node : Sequence->nodes()) {
      flipEmptyThenImpl(AST, Node);
    }
  } else if (auto *If = llvm::dyn_cast<IfNode>(Node)) {
    if (!If->hasThen()) {
      If->setThen(If->getElse());
      If->setElse(nullptr);

      // Invert the conditional expression of the current `IfNode`.
      UniqueExpr Not;
      revng_assert(If->getCondExpr());
      Not.reset(new NotNode(If->getCondExpr()));
      ExprNode *NotNode = AST.addCondExpr(std::move(Not));
      If->replaceCondExpr(NotNode);

      rc_recur flipEmptyThenImpl(AST, If->getThen());
    } else {

      // We are sure to have the `then` branch since the previous check did
      // not verify
      rc_recur flipEmptyThenImpl(AST, If->getThen());

      // We have not the same assurance for the `else` branch
      if (If->hasElse()) {
        rc_recur flipEmptyThenImpl(AST, If->getElse());
      }
    }
  } else if (auto *Scs = llvm::dyn_cast<ScsNode>(Node)) {
    if (Scs->hasBody())
      rc_recur flipEmptyThenImpl(AST, Scs->getBody());
  } else if (auto *Switch = llvm::dyn_cast<SwitchNode>(Node)) {

    for (auto &LabelCasePair : Switch->cases())
      rc_recur flipEmptyThenImpl(AST, LabelCasePair.second);
  }
}

void flipEmptyThen(ASTTree &AST, ASTNode *RootNode) {
  flipEmptyThenImpl(AST, RootNode);
}

static RecursiveCoroutine<ASTNode *> collapseSequencesImpl(ASTTree &AST,
                                                           ASTNode *Node) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);
    SequenceNode::links_container &SeqVec = Seq->getChildVec();

    // In place of a sequence node, we need just to inspect all the nodes in the
    // sequence.

    // In this support vector, we will place the index and the size for each
    // sequence replacement list.
    std::vector<std::pair<unsigned, unsigned>> ReplacementVector;

    // This index is used to keep track of all children sequence nodes.
    unsigned I = 0;
    unsigned TotalNestedChildren = 0;
    for (ASTNode *&N : Seq->nodes()) {
      N = rc_recur collapseSequencesImpl(AST, N);

      // After analyzing the node, we check if the node is a sequence node
      // itself. If that's the case, we annotate the fact, in order to collapse
      // them in the current sequence node after resizing the vector.
      if (auto *SubSeq = llvm::dyn_cast<SequenceNode>(N)) {
        ReplacementVector.push_back(std::make_pair(I, SubSeq->length()));
        TotalNestedChildren += SubSeq->length();
      }
      I++;
    }

    // Reserve the required size in the sequence node child vector, in order to
    // avoid excessive reallocations. In the computation of the required new
    // size, remember that for every sublist we add we actually need to subtract
    // one (the spot of the sub sequence node that is being removed, which will
    // now disappear and whose place will be taken by the first node of the
    // sublist).
    SeqVec.reserve(SeqVec.size() + TotalNestedChildren
                   - ReplacementVector.size());

    // Replace in the original sequence list the child sequence node with the
    // content of the node itself.

    // This offset is used to compute the relative position of successive
    // sequence node (with respect to their original position), once the vector
    // increases in size due to the previous insertions (we actually do a -1 to
    // keep into account the sequence node itself which is being replaced).
    unsigned Offset = 0;
    for (auto &Pair : ReplacementVector) {
      unsigned Index = Pair.first + Offset;
      unsigned VecSize = Pair.second;
      Offset += VecSize - 1;

      // The substitution is done by taking an iterator the the old sequence
      // node, erasing it from the node list vector of the parent sequence,
      // inserting the nodes of the collapsed sequence node, and then removing
      // it from the AST.
      auto InternalSeqIt = SeqVec.begin() + Index;
      auto *InternalSeq = llvm::cast<SequenceNode>(*InternalSeqIt);
      auto It = SeqVec.erase(InternalSeqIt);
      SeqVec.insert(It,
                    InternalSeq->nodes().begin(),
                    InternalSeq->nodes().end());
      AST.removeASTNode(InternalSeq);
    }
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Scs = llvm::cast<ScsNode>(Node);
    if (Scs->hasBody()) {
      ASTNode *Body = Scs->getBody();
      ASTNode *NewBody = rc_recur collapseSequencesImpl(AST, Body);
      Scs->setBody(NewBody);
    }
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // First of all, we recursively invoke the analysis on the children of the
    // `IfNode` (we discussed and said that further simplifications down in
    // the AST do not alter the `nofallthrough property`).
    if (If->hasThen()) {

      // We only have a `then` branch, proceed with the recursive visit.
      ASTNode *Then = If->getThen();
      ASTNode *NewThen = rc_recur collapseSequencesImpl(AST, Then);
      If->setThen(NewThen);
    }
    if (If->hasElse()) {

      // We only have a `else` branch, proceed with the recursive visit.
      ASTNode *Else = If->getElse();
      ASTNode *NewElse = rc_recur collapseSequencesImpl(AST, Else);
      If->setElse(NewElse);
    }
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);
    for (auto &LabelCasePair : Switch->cases())
      LabelCasePair
        .second = rc_recur collapseSequencesImpl(AST, LabelCasePair.second);
  } break;
  case ASTNode::NK_Code:
  case ASTNode::NK_Set:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break:
    // Do nothing.
    break;
  default:
    revng_unreachable();
  }
  rc_return Node;
}

ASTNode *collapseSequences(ASTTree &AST, ASTNode *RootNode) {
  return collapseSequencesImpl(AST, RootNode);
}

// Helper function which simplifies sequence nodes composed by a single AST
// node.
static RecursiveCoroutine<ASTNode *> simplifyAtomicSequenceImpl(ASTTree &AST,
                                                                ASTNode *Node) {
  switch (Node->getKind()) {

  case ASTNode::NK_List: {
    auto *Sequence = llvm::cast<SequenceNode>(Node);
    switch (Sequence->length()) {

    case 0:
      Node = nullptr;

      // Actually remove the sequence node from the ASTTree.
      AST.removeASTNode(Sequence);
      break;

    case 1:
      Node = rc_recur simplifyAtomicSequenceImpl(AST, Sequence->getNodeN(0));

      // Actually remove the sequence node from the ASTTree.
      AST.removeASTNode(Sequence);
      break;

    default:
      bool Empty = true;
      for (ASTNode *&Node : Sequence->nodes()) {
        Node = rc_recur simplifyAtomicSequenceImpl(AST, Node);
        if (nullptr != Node)
          Empty = false;
      }
      revng_assert(not Empty);
    }
  } break;

  case ASTNode::NK_If: {
    auto *If = llvm::cast<IfNode>(Node);

    if (If->hasThen())
      If->setThen(rc_recur simplifyAtomicSequenceImpl(AST, If->getThen()));

    if (If->hasElse())
      If->setElse(rc_recur simplifyAtomicSequenceImpl(AST, If->getElse()));

  } break;

  case ASTNode::NK_Switch: {

    auto *Switch = llvm::cast<SwitchNode>(Node);

    // In case the recursive call to `simplifyAtomicSequence` gives origin to a
    // complete simplification of the default node of the switch, setting its
    // corresponding `ASTNode` to `nullptr` already does the job, since having
    // the corresponding `Default` field set to `nullptr` means that the switch
    // node has no default.
    auto LabelCasePairIt = Switch->cases().begin();
    auto LabelCasePairEnd = Switch->cases().end();
    while (LabelCasePairIt != LabelCasePairEnd) {
      auto *NewCaseNode = rc_recur
        simplifyAtomicSequenceImpl(AST, LabelCasePairIt->second);
      if (nullptr == NewCaseNode) {
        if (not Switch->hasDefault()) {
          LabelCasePairIt = Switch->cases().erase(LabelCasePairIt);
          LabelCasePairEnd = Switch->cases().end();
        } else {
          LabelCasePairIt->second = AST.addSwitchBreak(Switch);
        }
      } else {
        LabelCasePairIt->second = NewCaseNode;
        ++LabelCasePairIt;
      }
    }

  } break;

  case ASTNode::NK_Scs: {
    auto *Scs = llvm::cast<ScsNode>(Node);
    if (Scs->hasBody())
      Scs->setBody(rc_recur simplifyAtomicSequenceImpl(AST, Scs->getBody()));
  } break;

  case ASTNode::NK_Code:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Set:
    // Do nothing
    break;

  default:
    revng_unreachable();
  }

  rc_return Node;
}

ASTNode *simplifyAtomicSequence(ASTTree &AST, ASTNode *RootNode) {
  return simplifyAtomicSequenceImpl(AST, RootNode);
}

/// Beautify helper function that is used to enforce the principal invariants
/// that we want in the _normalized_ version of the GHAST.
/// Mainly, the objective of this function is to:
/// 1) Transform an `IfNode` with an empth `then` branch and non-empty `else`,
///    to an equivalent `IfNode` with the condition _flipped_, and the `then`
///    and `else` branches swapped.
/// 2) Enforce that no `SequenceNode` has as a direct child, another
///   `SequenceNode`. In this situation, we want to _copy_ all the children of
///    the nested `SequenceNode`, in place of the node itself in the parent
///    `SequenceNode`.
/// 3) Enforce that no `SequenceNode` with a single child are present. To do
///    that, we elide the `SequenceNode` altogether, and move the unique child
///    to the parent of the `SequenceNode`.
ASTNode *canonicalize(ASTTree &AST, ASTNode *RootNode) {
  flipEmptyThen(AST, RootNode);
  RootNode = collapseSequences(AST, RootNode);
  RootNode = simplifyAtomicSequence(AST, RootNode);
  AST.setRoot(RootNode);

  return RootNode;
}
