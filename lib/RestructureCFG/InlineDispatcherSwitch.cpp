/// \file InlineDispatcherSwitch.cpp
/// Beautification pass to inline dispatcher switch cases where a case of the
/// switch is inlinable in a single location in the loop
///

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <algorithm>

#include "llvm/ADT/SetOperations.h"
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
#include "revng-c/Support/FunctionTags.h"

#include "FallThroughScopeAnalysis.h"
#include "InlineDispatcherSwitch.h"

using namespace llvm;

// This map is used to: 1) keep the mapping between each exit dispatcher and the
// corresponding loop 2) A boolean value which tells us if the `SetNode` should
// be remove during the inlining of the case
using LoopDispatcherMap = std::map<SwitchNode *, std::pair<ScsNode *, bool>>;

using NodeSet = llvm::SmallSet<SetNode *, 1>;
using SetNodeCounterMap = std::map<uint64_t, NodeSet>;

static RecursiveCoroutine<void>
countSetNodeInLoop(ASTNode *Node, SetNodeCounterMap &CounterMap) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // In place of a sequence node, we need just to inspect all the nodes in the
    // sequence
    for (ASTNode *&N : Seq->nodes()) {
      rc_recur countSetNodeInLoop(N, CounterMap);
    }
  } break;
  case ASTNode::NK_Scs: {
    // We do not traverse loops, we are only insterested into `SetNode`s of the
    // current loop
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // Inspect the `then` and `else` branches
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      rc_recur countSetNodeInLoop(Then, CounterMap);
    }
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      rc_recur countSetNodeInLoop(Else, CounterMap);
    }
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);

    // First of all, we recursively process the `case` nodes contained in the
    // `switch` in order to process the inner portion of the AST
    for (auto &LabelCasePair : Switch->cases()) {
      rc_recur countSetNodeInLoop(LabelCasePair.second, CounterMap);
    }
  } break;
  case ASTNode::NK_Set: {
    auto *Set = llvm::cast<SetNode>(Node);

    // We are only interested in counting `SetNode`s related to exit dispatchers
    using DispatcherKind = typename SetNode::DispatcherKind;
    if (Set->getDispatcherKind() == DispatcherKind::DK_Exit) {
      CounterMap[Set->getStateVariableValue()].insert(Set);
    }
  } break;

  case ASTNode::NK_Code:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break:
    // Do nothing
    break;
  default:
    revng_unreachable();
  }

  rc_return;
}

static RecursiveCoroutine<ASTNode *> addToDispatcherSet(ASTTree &AST,
                                                        ASTNode *Node,
                                                        ASTNode *InlinedBody,
                                                        ASTNode *InlinedSet,
                                                        bool RemoveSetNode) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // In place of a sequence node, we need just to inspect all the nodes in the
    // sequence
    for (ASTNode *&N : Seq->nodes()) {
      N = rc_recur addToDispatcherSet(AST,
                                      N,
                                      InlinedBody,
                                      InlinedSet,
                                      RemoveSetNode);
    }

    // When a inlining inserts a `SwitchBreak` node, this is removed, so we need
    // to clean the `SequenceNode` here
    Seq->removeNode(nullptr);
  } break;
  case ASTNode::NK_Scs: {
    // We do not traverse loops, we are only insterested into `SetNode`s of the
    // current loop
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // Inspect the `then` and `else` branches
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      ASTNode *NewThen = rc_recur addToDispatcherSet(AST,
                                                     Then,
                                                     InlinedBody,
                                                     InlinedSet,
                                                     RemoveSetNode);
      If->setThen(NewThen);
    }
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      ASTNode *NewElse = rc_recur addToDispatcherSet(AST,
                                                     Else,
                                                     InlinedBody,
                                                     InlinedSet,
                                                     RemoveSetNode);
      If->setElse(NewElse);
    }
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);

    // First of all, we recursively process the `case` nodes contained in the
    // `switch` in order to process the inner portion of the AST
    for (auto &LabelCasePair : Switch->cases()) {
      LabelCasePair.second = rc_recur addToDispatcherSet(AST,
                                                         LabelCasePair.second,
                                                         InlinedBody,
                                                         InlinedSet,
                                                         RemoveSetNode);
    }
  } break;
  case ASTNode::NK_Set: {
    auto *Set = llvm::cast<SetNode>(Node);

    // We have reached the `SetNode` marked for the inlining
    if (Set == InlinedSet) {

      // The `SwitchBreakNode` should not reach this point, but handled in the
      // previous branch
      revng_assert(not llvm::dyn_cast_or_null<SwitchBreakNode>(InlinedBody));

      // Depending on the `RemoveSetNode` parameter, we may either remove the
      // `SetNode` and inline the `InlinedBody`, or prepend the `InlinedBody` to
      // the `SetNode`.
      // The `InlinedBody` parameter can be either a `nullptr`, in case we are
      // simplifying away a `SwitchBreak` node, or the actual `ASTNode *` which
      // is substituting the `SetNode`.
      if (RemoveSetNode) {
        rc_return InlinedBody;
      } else {
        if (InlinedBody) {
          // If we are inlining the `InlinedBody` and preserving the `SetNode`,
          // we need to create a new `SequenceNode` to group them together
          SequenceNode *Sequence = AST.addSequenceNode();
          Sequence->addNode(InlinedBody);
          Sequence->addNode(Set);
          rc_return Sequence;
        } else {
          // We don't have to inline anything, and we preserve the `SetNode`
          rc_return Set;
        }
      }
    }
  } break;

  case ASTNode::NK_Code:
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

static bool isDispatcherIf(ASTNode *If) {
  const ExprNode *E = llvm::cast<IfNode>(If)->getCondExpr();
  using NodeKind = ExprNode::NodeKind;
  if (E->getKind() == NodeKind::NK_LoopStateCompare) {
    return true;
  }
  return false;
}

static LoopDispatcherMap computeLoopDispatcher(ASTTree &AST) {
  LoopDispatcherMap ResultMap;

  for (ASTNode *Node : AST.nodes()) {
    if (auto *Sequence = llvm::dyn_cast<SequenceNode>(Node)) {
      ASTNode *PrevNode = nullptr;
      for (ASTNode *N : Sequence->nodes()) {

        // If we encounter a dispatcher switch, we should assert that the
        // previous node is a `ScsNode` (whose exit dispatcher the current
        // switch is of), and save it in the map
        if (auto *Switch = llvm::dyn_cast<SwitchNode>(N)) {

          // We handle weaved switches only as children of the original `switch`
          // (couple of lines after), where we can actually retrieve the related
          // loop
          if (Switch->isWeaved()) {
            continue;
          }

          // A `SwitchNode` is a dispatcher switch if it has no associated
          // condition
          using DispatcherKind = typename SwitchNode::DispatcherKind;
          if (not Switch->getCondition()
              and Switch->getDispatcherKind() == DispatcherKind::DK_Exit) {
            ScsNode *RelatedLoop = llvm::cast<ScsNode>(PrevNode);
            ResultMap[Switch] = std::make_pair(RelatedLoop, true);

            // We also need to annotate the mapping for weaved switches, that
            // are contained in the `SwitchNode` under current analysis. Since
            // there may be multiple levels of weaved dispatchers, we need to
            // perform this collection iteratively.
            llvm::SmallVector<SwitchNode *> SwitchQueue;
            SwitchQueue.push_back(Switch);
            while (not SwitchQueue.empty()) {
              SwitchNode *SubSwitch = SwitchQueue.back();
              SwitchQueue.pop_back();

              for (const auto &[LabelSet, Case] : SubSwitch->cases()) {
                revng_assert(LabelSet.size() > 0);

                // A weaved `SwitchNode` can be present only inside `case`s with
                // multiple labels
                if (LabelSet.size() > 1) {
                  SwitchNode *WeavedSwitch = nullptr;
                  bool RemoveSetNode;

                  // We have two possible situations:
                  // 1) The weaved `SwitchNode` is a direct child of the parent
                  //    `SwitchNode`. In this case, when we perform the inline
                  //    of the `case`s, we should remove the `SetNode`.
                  // 2) The weaved `SwitchNode` is the first node of a
                  //    `SequenceNode`. In this case, when we perform the inline
                  //    of the `case`s, we should leave the `SetNode`, because
                  //    the remaining nodes of the `SequenceNode` will not be
                  //    simplified by the inlining.
                  if (llvm::isa<SwitchNode>(Case)) {
                    WeavedSwitch = llvm::cast<SwitchNode>(Case);
                    RemoveSetNode = true;
                  } else if (auto *Seq = llvm::dyn_cast<SequenceNode>(Case)) {
                    WeavedSwitch = llvm::cast<SwitchNode>(Seq->getNodeN(0));
                    RemoveSetNode = false;
                  } else {
                    revng_abort("Unexpected Weaved SwitchNode");
                  }

                  revng_assert(WeavedSwitch->isWeaved());
                  ResultMap[WeavedSwitch] = std::make_pair(RelatedLoop,
                                                           RemoveSetNode);
                  SwitchQueue.push_back(WeavedSwitch);
                }
              }
            }
          }
        } else if (auto *If = llvm::dyn_cast<IfNode>(N)) {

          // The current assumption is that the current `InlineDispatcherSwitch`
          // pass runs before `switch`es with one or two cases are promoted to
          // `if`s. If this is not true, we need to adapt this pass.
          revng_assert(not isDispatcherIf(If));
        }

        // At each iteration, assign the `PrevNode` variable
        PrevNode = N;
      }
    }
  }

  return ResultMap;
}

static RecursiveCoroutine<bool> containsSet(ASTNode *Node) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    bool HasSet = false;
    for (ASTNode *N : Seq->nodes()) {
      HasSet = HasSet or rc_recur containsSet(N);
    }
    rc_return HasSet;
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Loop = llvm::cast<ScsNode>(Node);

    if (Loop->hasBody()) {
      ASTNode *Body = Loop->getBody();
      rc_return containsSet(Body);
    } else {
      rc_return false;
    }
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    bool HasSet = false;
    // Inspect the `then` and `else` branches
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      HasSet = HasSet or rc_recur containsSet(Then);
    }
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      HasSet = HasSet or rc_recur containsSet(Else);
    }

    rc_return HasSet;
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);

    bool HasSet = false;
    // First of all, we recursively process the `case` nodes contained in the
    // `switch` in order to process the inner portion of the AST
    for (auto &LabelCasePair : Switch->cases()) {
      HasSet = HasSet or rc_recur containsSet(LabelCasePair.second);
    }

    rc_return HasSet;
  } break;
  case ASTNode::NK_Set: {
    rc_return true;
  } break;

  case ASTNode::NK_Code:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break: {
    rc_return false;
  } break;
  default:
    revng_unreachable();
  }

  revng_abort();
}

static void processNestedWeavedSwitches(SwitchNode *Switch) {
  // If a exit dispatcher weaved node was a child of the current exit
  // dispatcher, it may be that:
  // 1) The inlining took place for some of the cases, so we need to take
  //    care of removing those labels from the parent case containing them.
  // 2) The weaved switch disappeared entirely, so we need to remove the
  //    parent case entirely.
  std::set<size_t> ToRemoveCaseIndex;
  for (auto &Group : llvm::enumerate(Switch->cases())) {
    unsigned Index = Group.index();
    auto &[LabelSet, Case] = Group.value();

    // In case of a weaved switch which is nested in this one, we may have
    // that one of the cases has been simplified to `nullptr`, therefore, we
    // should take care of removing it
    if (Case == nullptr) {
      ToRemoveCaseIndex.insert(Index);
      continue;
    }

    revng_assert(LabelSet.size() != 0);

    // We are interested in searching for immediately nested weaved
    // `SwitchNode`s, which are the only ones that we are able to
    // handle in this step.
    // The other possibility, is that the nested weaved `SwitchNode` is the
    // first node in a `SequenceNode`. In that situation, we may inline the
    // content of the cases of the nested weaved `SwitchNode`, but we still need
    // to retain the `case`s in the parent `SwitchNode`, because the nodes in
    // the `SequenceNode` still need to be _executed_ when the `loop_state_var`
    // assumes the current values.
    if (LabelSet.size() > 1 and llvm::isa<SwitchNode>(Case)) {
      auto *WeavedSwitch = llvm::cast<SwitchNode>(Case);
      llvm::SmallSet<uint64_t, 1> WeavedLabels;
      for (const auto &[WeavedLabelSet, WeavedCase] : WeavedSwitch->cases()) {
        WeavedLabels.insert(WeavedLabelSet.begin(), WeavedLabelSet.end());
      }

      // Keep only the intersection between the original labels and the
      // ones found in the weaved switch
      for (auto It = LabelSet.begin(); It != LabelSet.end();) {
        const auto &E = *It;
        ++It;
        if (not WeavedLabels.contains(E)) {
          LabelSet.erase(E);
        }
      }

      // Additional simplification step, that takes care of removing a
      // nested `WeavedSwitch` If, after the processing, the only remaining
      // case in the weaved child switch, is equivalent to the current case
      // in the parent switch, we can simplify it by removing the weaved
      // switch altogether
      if (WeavedSwitch->cases_size() == 1) {
        revng_assert(LabelSet == WeavedLabels);
        Case = WeavedSwitch->cases().begin()->second;
      }
    }
  }

  for (auto ToRemoveCase : llvm::reverse(ToRemoveCaseIndex)) {
    Switch->removeCaseN(ToRemoveCase);
  }
}

static RecursiveCoroutine<ASTNode *>
inlineDispatcherSwitchImpl(ASTTree &AST,
                           ASTNode *Node,
                           const LoopDispatcherMap &LoopDispatcherMap) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // In place of a sequence node, we need just to inspect all the nodes in the
    // sequence
    for (ASTNode *&N : Seq->nodes()) {
      N = rc_recur inlineDispatcherSwitchImpl(AST, N, LoopDispatcherMap);
    }

    // In this beautify, it may be that a dispatcher switch is completely
    // removed, therefore leaving a `nullptr` in a `SequenceNode`. We remove all
    // these `nullptr` from the `SequenceNode` after the processing has taken
    // place
    Seq->removeNode(nullptr);

  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Scs = llvm::cast<ScsNode>(Node);

    // Inspect loop nodes
    if (Scs->hasBody()) {
      ASTNode *Body = Scs->getBody();
      ASTNode *NewBody = rc_recur inlineDispatcherSwitchImpl(AST,
                                                             Body,
                                                             LoopDispatcherMap);
      Scs->setBody(NewBody);
    }

  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // Inspect the `then` and `else` branches
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      ASTNode *NewThen = rc_recur inlineDispatcherSwitchImpl(AST,
                                                             Then,
                                                             LoopDispatcherMap);
      If->setThen(NewThen);
    }
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      ASTNode *NewElse = rc_recur inlineDispatcherSwitchImpl(AST,
                                                             Else,
                                                             LoopDispatcherMap);
      If->setElse(NewElse);
    }
  } break;
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);

    // First of all, we recursively process the `case` nodes contained in the
    // `switch` in order to process the inner portion of the AST
    for (auto &LabelCasePair : Switch->cases()) {
      LabelCasePair
        .second = rc_recur inlineDispatcherSwitchImpl(AST,
                                                      LabelCasePair.second,
                                                      LoopDispatcherMap);
    }

    // Execute the promotion routine only for dispatcher switches
    using DispatcherKind = typename SwitchNode::DispatcherKind;
    if (not Switch->getCondition()
        and Switch->getDispatcherKind() == DispatcherKind::DK_Exit) {

      // Process the nested weaved switches, if present in the current `Switch`
      processNestedWeavedSwitches(Switch);

      // Precompute how many `SetNode`s are present in the related `Scs`
      auto &[RelatedLoop, RemoveSetNode] = LoopDispatcherMap.at(Switch);
      SetNodeCounterMap SetCounterMap;
      countSetNodeInLoop(RelatedLoop->getBody(), SetCounterMap);

      // The inlining routine should proceed as follows:
      // 1) In the first phase, we iterate through all the cases, and try to
      //    inline the body of the case in place of the corresponding
      //    `SetNode`, if there is a single one (this criterion ensures that
      //    we are not introducing duplication in the beautify phase), and the
      //    body of the case is a `GenericNoFallThrough` scope.
      // 2) Additionally, if in the previous step, we were able to inline all
      //    the cases of the switch, we can additionally remove entirely the
      //    dispatcher switch.
      std::set<size_t> ToRemoveCaseIndex;
      for (auto &Group : llvm::enumerate(Switch->cases())) {
        unsigned Index = Group.index();
        auto &[LabelSet, Case] = Group.value();

        // Here we handle only cases with a single label. The multiple labels
        // are handled in the recursive step
        if (LabelSet.size() == 1) {
          auto Label = *LabelSet.begin();
          auto &Sets = SetCounterMap.at(Label);
          if (llvm::isa<SwitchBreakNode>(Case)) {

            // If the body of the case is composed by a single `SwitchBreakNode`
            // node, we can remove it, by virtually inlining a `nullptr` in the
            // place of all the corresponding `SetNode`s (we can do it multiple
            // times since we are not duplicating code here)
            for (SetNode *Set : Sets) {
              addToDispatcherSet(AST,
                                 RelatedLoop->getBody(),
                                 nullptr,
                                 Set,
                                 RemoveSetNode);
            }
            ToRemoveCaseIndex.insert(Index);
          } else if (Sets.size() == 1
                     and (not needsLoopVar(RelatedLoop)
                          or not containsSet(Case))) {

            // We inline the body of the case only if the following conditions
            // stand:
            // 1) There is a single `SetNode` for the corresponding label in the
            // body of the loop where we inline the case.
            // 2) Either, the loop where we inline the case does not require a
            // `state-variable`, or if it requires it, the body of the case we
            // inline does not contain any `SetNode`. In that case indeed, we
            // would be moving a `SetNode` from a scope to another one, which is
            // not semantics preserving.
            addToDispatcherSet(AST,
                               RelatedLoop->getBody(),
                               Case,
                               *Sets.begin(),
                               RemoveSetNode);
            ToRemoveCaseIndex.insert(Index);
          }
        }
      }

      // We remove the cases from the last to the first (we avoid invalidating
      // the elements in the underlying `llvm::SmallVector`)
      for (auto ToRemoveCase : llvm::reverse(ToRemoveCaseIndex)) {
        Switch->removeCaseN(ToRemoveCase);
      }

      // Finally, if we end up with a `SwitchNode` with no remaining cases, we
      // should completely remove it. We do that by returning a `nullptr`.
      if (Switch->cases_size() == 0) {
        rc_return nullptr;
      }
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

ASTNode *inlineDispatcherSwitch(ASTTree &AST, ASTNode *RootNode) {

  // Compute the loop -> dispatcher switch correspondence
  LoopDispatcherMap LoopDispatcherMap = computeLoopDispatcher(AST);

  RootNode = inlineDispatcherSwitchImpl(AST, RootNode, LoopDispatcherMap);

  // Update the root field of the AST
  AST.setRoot(RootNode);

  return RootNode;
}
