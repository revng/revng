/// \file InlineDispatcherSwitch.cpp
/// Beautification pass to inline dispatcher switch cases where a case of the
/// switch is inlinable in a single location in the loop
///

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>

#include "llvm/ADT/SetOperations.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/RestructureCFG/ASTNode.h"
#include "revng/RestructureCFG/ASTNodeUtils.h"
#include "revng/RestructureCFG/ASTTree.h"
#include "revng/RestructureCFG/ExprNode.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"

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

    // Perform the inlining over all the cases
    llvm::SmallVector<size_t> ToRemoveCaseIndex;
    for (auto &Group : llvm::enumerate(Switch->cases())) {
      unsigned Index = Group.index();
      auto &LabelCasePair = Group.value();
      LabelCasePair.second = rc_recur addToDispatcherSet(AST,
                                                         LabelCasePair.second,
                                                         InlinedBody,
                                                         InlinedSet,
                                                         RemoveSetNode);

      if (LabelCasePair.second == nullptr) {
        ToRemoveCaseIndex.push_back(Index);
      }
    }

    // The inlining step may remove the body of some case, in such situation we
    // need to remove the case from the switch (leaving the `nullptr` breaks the
    // semantics of the AST)
    for (auto ToRemoveCase : llvm::reverse(ToRemoveCaseIndex)) {
      Switch->removeCaseN(ToRemoveCase);
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

/// Wrapper helper used to remove a specific `SetNode` from the body of a loop,
/// reusing the code of the `addToDispatcherSet` helper
static void removeDispatcherSet(ASTTree &AST, ASTNode *Node, ASTNode *Set) {

  // We perform the `SetNode` by instructing `addToDispatcherSet` to inline a
  // `nullptr`, and to remove the original `SetNode`
  addToDispatcherSet(AST, Node, nullptr, Set, true);
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

      // We use the `LastSeenLoop` variable to propagate the last encountered
      // loop in a sequence, so that we can process also weaved `switch`es that
      // are not directly nested in their main correspondent `switch`
      ScsNode *LastSeenLoop = nullptr;
      for (ASTNode *N : Sequence->nodes()) {
        if (auto *Switch = llvm::dyn_cast<SwitchNode>(N)) {

          // A `SwitchNode` is a dispatcher switch if it has no associated
          // condition
          using DispatcherKind = typename SwitchNode::DispatcherKind;
          if (not Switch->getCondition()
              and Switch->getDispatcherKind() == DispatcherKind::DK_Exit) {
            ScsNode *RelatedLoop = nullptr;

            // When we encounter a weaved `switch` in a sequence, since in
            // this exploration we are just looking at all the nodes in the
            // AST without doing a recursive visit, 2 situations are possible:
            // 1) The weaved `switch` is nested inside the main `switch` it
            // refers to, and we will eventually assign its related loop when
            // we visit such main `switch`.
            // 2) We encounter a weaved `switch` which is not nested
            // inside its main `switch`, in this case in the current sequence
            // we are exploring we should have visited the its related loop.
            if (Switch->isWeaved()) {
              if (LastSeenLoop) {
                // We are in situation 2)
                RelatedLoop = LastSeenLoop;

                // We also propagate the `LastSeenLoop` to the next weaved in
                // the sequence (if present), not overwriting `LastSeenLoop`
              } else {
                // We are in situation 1)

                // We blank `LastSeenLoop` (this is superfluous)
                LastSeenLoop = nullptr;

                continue;
              }
            } else {
              // In occurrence of a non-weaved `switch`, we know that the
              // related loop is the previous node in the sequence
              RelatedLoop = llvm::cast<ScsNode>(PrevNode);

              // From this point on, we also set the `LastSeenLoop` to the
              // `RelatedLoop`, for eventual weaved `switch`es which may follow
              // the current main `switch`
              LastSeenLoop = RelatedLoop;
            }
            revng_assert(RelatedLoop);
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
                  llvm::SmallVector<SwitchNode *> WeavedSwitches;
                  bool RemoveSetNode;

                  // We have three possible situations:
                  // 1) The weaved `SwitchNode` is a direct child of the parent
                  //    `SwitchNode`. In this case, when we perform the inline
                  //    of the `case`s, we should remove the `SetNode`.
                  // 2) We have (multiple) weaved `SwitchNode`s inside a
                  //    `SequenceNode`. The nodes we are interested in, are the
                  //    first n consecutive nodes.
                  // 3) The weaved `SwitchNode` is not nested inside its parent
                  //    `SwitchNode`, but it follows it in a lexicographic
                  //    sense.
                  if (llvm::isa<SwitchNode>(Case)) {
                    SwitchNode *WeavedSwitch = llvm::cast<SwitchNode>(Case);
                    revng_assert(WeavedSwitch->isWeaved());
                    WeavedSwitches.push_back(WeavedSwitch);
                    RemoveSetNode = true;
                  } else if (auto *Seq = llvm::dyn_cast<SequenceNode>(Case)) {
                    RemoveSetNode = true;
                    for (ASTNode *N : Seq->nodes()) {
                      SwitchNode *WeavedSwitch = llvm::dyn_cast<SwitchNode>(N);
                      if (WeavedSwitch and WeavedSwitch->isWeaved()) {
                        WeavedSwitches.push_back(WeavedSwitch);
                      } else {

                        // As soon as we find an `ASTNode` that is not a weaved
                        // `switch`, we exit from the routine, and set the
                        // `RemoveSetNode` to false (indeed, if the `sequence`
                        // containing the sub-weaved-`switch` contains also some
                        // other nodes in the back, it means that we should not
                        // need to remove the `SetNode` to preserve the
                        // semantics, since we cannot inline completely the body
                        // of the `case`)
                        RemoveSetNode = false;
                        break;
                      }
                    }
                  } else {
                    continue;
                  }

                  for (SwitchNode *WeavedSwitch : WeavedSwitches) {
                    revng_assert(WeavedSwitch->isWeaved());
                    ResultMap[WeavedSwitch] = std::make_pair(RelatedLoop,
                                                             RemoveSetNode);
                    SwitchQueue.push_back(WeavedSwitch);
                  }
                }
              }
            }
          }
        } else {

          // We reset the `LastSeenLoop` at each node except for each main
          // `switch` that may be followed by a corresponding weaved `switch`
          LastSeenLoop = nullptr;

          // The current assumption is that the current `InlineDispatcherSwitch`
          // pass runs before `switch`es with one or two cases are promoted to
          // `if`s. Therefore no dispatcher `if` should be present at this
          // stage. If this is not true, we need to adapt this pass.
          revng_assert(not(llvm::isa<IfNode>(N) and isDispatcherIf(N)));
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
  }
  case ASTNode::NK_Scs: {
    ScsNode *Loop = llvm::cast<ScsNode>(Node);

    if (Loop->hasBody()) {
      ASTNode *Body = Loop->getBody();
      rc_return containsSet(Body);
    } else {
      rc_return false;
    }
  }
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
  }
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);

    bool HasSet = false;
    // First of all, we recursively process the `case` nodes contained in the
    // `switch` in order to process the inner portion of the AST
    for (auto &LabelCasePair : Switch->cases()) {
      HasSet = HasSet or rc_recur containsSet(LabelCasePair.second);
    }

    rc_return HasSet;
  }
  case ASTNode::NK_Set: {
    rc_return true;
  }

  case ASTNode::NK_Code:
  case ASTNode::NK_SwitchBreak:
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break: {
    rc_return false;
  }
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
  llvm::SmallVector<size_t> ToRemoveCaseIndex;
  for (auto &Group : llvm::enumerate(Switch->cases())) {
    unsigned Index = Group.index();
    auto &[LabelSet, Case] = Group.value();

    // In case of a weaved switch which is nested in this one, we may have
    // that one of the cases has been simplified to `nullptr`, therefore, we
    // should take care of removing it
    if (Case == nullptr) {
      ToRemoveCaseIndex.push_back(Index);
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
    // the `SequenceNode` still need to be *executed* when the `loop_state_var`
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

static RecursiveCoroutine<bool> containsContinueOrBreak(ASTNode *Node) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // As soon as we find an element of the `SequenceNode` that contains non
    // local CF, we can signal it by early exiting returning `true`
    for (ASTNode *N : llvm::reverse(Seq->nodes())) {
      if (rc_recur containsContinueOrBreak(N)) {
        rc_return true;
      }
    }

    // This will be executed only if no non local CF has been found in the
    // above iterations
    rc_return false;
  }
  case ASTNode::NK_Scs: {

    // At the current stage, we do not consider the possibility of inlining a
    // `ScsNode`, because it would mean envisioning a way to handle also an
    // eventual exit dispatcher associated to the `ScsNode`.
    // We may want, however, to enable this possibility in the future.
    rc_return true;
  }
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // We can early exit as soon as we find a branch with non local CF
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      if (rc_recur containsContinueOrBreak(Then)) {
        rc_return true;
      }
    }
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      if (rc_recur containsContinueOrBreak(Else)) {
        rc_return true;
      }
    }

    // This will be executed only if no non local CF has been found in both the
    // `then` and `else` branches
    rc_return false;
  }
  case ASTNode::NK_Switch: {
    auto *Switch = llvm::cast<SwitchNode>(Node);

    // We can early exit as soon as we find a `case` with non local CF
    for (auto &LabelCasePair : Switch->cases()) {
      if (rc_recur containsContinueOrBreak(LabelCasePair.second)) {
        rc_return true;
      }
    }

    // This will be executed only if no non local CF has been found in any of
    // the `case`s of the `switch`
    rc_return false;
  }
  case ASTNode::NK_Set:
  case ASTNode::NK_Code:
  case ASTNode::NK_SwitchBreak: {

    // The goal of this helper function is to identify just `continue` and
    // `break` statements related to loops. The `SwitchBreak` statement does not
    // fall in this category.
    rc_return false;
  }
  case ASTNode::NK_Continue:
  case ASTNode::NK_Break: {
    rc_return true;
  }
  default:
    revng_unreachable();
  }

  revng_abort();
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
    llvm::SmallVector<size_t> ToRemoveCaseIndex;
    for (auto &Group : llvm::enumerate(Switch->cases())) {
      unsigned Index = Group.index();
      auto &LabelCasePair = Group.value();
      LabelCasePair
        .second = rc_recur inlineDispatcherSwitchImpl(AST,
                                                      LabelCasePair.second,
                                                      LoopDispatcherMap);

      if (LabelCasePair.second == nullptr) {
        ToRemoveCaseIndex.push_back(Index);
      }
    }

    for (auto ToRemoveCase : llvm::reverse(ToRemoveCaseIndex)) {
      Switch->removeCaseN(ToRemoveCase);
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

            // The superfluous `SwitchBreak` removal should have been already
            // performed in the `simplifySwitchBreak` phase
            revng_abort();
          } else if (Sets.size() == 1
                     and (not needsLoopVar(RelatedLoop)
                          or not containsSet(Case))
                     and not containsContinueOrBreak(Case)) {

            // We inline the body of the case only if the following conditions
            // stand:
            // 1) There is a single `SetNode` for the corresponding label in the
            // body of the loop where we inline the case.
            // 2) Either, the loop where we inline the case does not require a
            // `state-variable`, or if it requires it, the body of the case we
            // inline does not contain any `SetNode`. In that case indeed, we
            // would be moving a `SetNode` from a scope to another one, which is
            // not semantics preserving.
            // 3) The `case` body we are trying to inline does not contain any
            // loop related `NonLocalControlFlow`, meaning no `break` or
            // `continue` statements. If that was the case, we would be moving
            // such statements from an external loop to a more nested one,
            // breaking the semantics.
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
    }

    // Finally, if we end up with a `SwitchNode` with no remaining cases, we
    // should completely remove it. We do that by returning a `nullptr`.
    if (Switch->cases_size() == 0) {
      rc_return nullptr;
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

ASTNode *inlineDispatcherSwitch(ASTTree &AST) {

  // Compute the loop -> dispatcher switch correspondence
  LoopDispatcherMap LoopDispatcherMap = computeLoopDispatcher(AST);

  ASTNode *RootNode = inlineDispatcherSwitchImpl(AST,
                                                 AST.getRoot(),
                                                 LoopDispatcherMap);

  // Update the root field of the AST
  AST.setRoot(RootNode);

  return RootNode;
}

static RecursiveCoroutine<ASTNode *>
simplifySwitchBreakImpl(ASTTree &AST,
                        ASTNode *Node,
                        const LoopDispatcherMap &LoopDispatcherMap) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // In place of a sequence node, we just need to inspect all the nodes in the
    // sequence
    for (ASTNode *&N : Seq->nodes()) {
      N = rc_recur simplifySwitchBreakImpl(AST, N, LoopDispatcherMap);
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
      ASTNode *NewBody = rc_recur simplifySwitchBreakImpl(AST,
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
      ASTNode *NewThen = rc_recur simplifySwitchBreakImpl(AST,
                                                          Then,
                                                          LoopDispatcherMap);
      If->setThen(NewThen);
    }
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      ASTNode *NewElse = rc_recur simplifySwitchBreakImpl(AST,
                                                          Else,
                                                          LoopDispatcherMap);
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
      LabelCasePair.second = rc_recur
        simplifySwitchBreakImpl(AST, LabelCasePair.second, LoopDispatcherMap);

      if (LabelCasePair.second == nullptr) {
        ToRemoveCaseIndex.push_back(Index);
      }
    }

    for (auto ToRemoveCase : llvm::reverse(ToRemoveCaseIndex)) {
      Switch->removeCaseN(ToRemoveCase);
    }

    // 1: if the `default` `case` is a `SwitchBreak`, we can simplify it away
    ASTNode *SwitchDefault = Switch->getDefault();
    if (SwitchDefault) {
      if (llvm::isa<SwitchBreakNode>(SwitchDefault)) {
        Switch->removeDefault();
      }
    }

    // 2: We can perform the `SwitchBreak` simplification only for `switch`es
    // that do not have a `default` `case`, because in such situation, removing
    // a `case` containing a single `Switchbreak` would not invalidate the
    // semantics. This is done after the previous simplification, because some
    // new opportunities may be unlocked by it.
    if (not Switch->hasDefault()) {

      // Search for cases that are composed by a single `SwitchBreak` node
      llvm::SmallVector<size_t> ToRemoveCaseIndex;

      // Save the `LabelSet`s associated to the removed `case`s
      llvm::SmallSet<size_t, 1> RemovedLabels;

      // We do not need to skip the `default` case here, because there is no
      // `default` in first place
      for (auto &Group : llvm::enumerate(Switch->cases())) {
        unsigned Index = Group.index();
        auto &[LabelSet, Case] = Group.value();

        if (llvm::isa<SwitchBreakNode>(Case)) {
          ToRemoveCaseIndex.push_back(Index);

          // If we are removing an atomic `Label`, we save it for later removal
          // of the associated `SetNode`. We do not remove the `SetNode`s
          // associated to multiple `Label`s `case`s, because that would not be
          // correct. The removal of the associated `SetNode` should be
          // performed only in correspondence of the `case`s containing the
          // atomic `Label`.
          if (LabelSet.size() == 1) {
            RemovedLabels.insert(*LabelSet.begin());
          }
        }
      }

      for (auto ToRemoveCase : llvm::reverse(ToRemoveCaseIndex)) {
        Switch->removeCaseN(ToRemoveCase);
      }

      // In case we are simplifying the `SwitchBreak` associated to an exit
      // dispatcher, we also need to remove the corresponding `SetNode`s
      using DispatcherKind = typename SwitchNode::DispatcherKind;
      if (not Switch->getCondition()
          and Switch->getDispatcherKind() == DispatcherKind::DK_Exit) {

        // Precompute how many `SetNode`s are present in the related `Scs`
        auto &[RelatedLoop, RemoveSetNode] = LoopDispatcherMap.at(Switch);
        SetNodeCounterMap SetCounterMap;
        countSetNodeInLoop(RelatedLoop->getBody(), SetCounterMap);

        // Remove the `SetNode`s associated to the removed `Label`s
        for (auto &Label : RemovedLabels) {
          auto &Sets = SetCounterMap.at(Label);
          for (auto *Set : Sets) {
            removeDispatcherSet(AST, RelatedLoop->getBody(), Set);
          }
        }
      }
    }

    // If our beautify pass has removed all the cases, we should return
    // `nulltpr` to signal this fact
    if (Switch->cases_size() == 0) {
      rc_return nullptr;
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

/// This simplification routine is aimed at removing `SwitchBreak` nodes that
/// are superfluous. We consider such nodes as `SwitchBreakNode`s that compose
/// the entirety of a `case` of a `switch`, which must not have a `default`
/// `case`.
ASTNode *simplifySwitchBreak(ASTTree &AST) {

  // Compute the loop -> dispatcher switch correspondence
  LoopDispatcherMap LoopDispatcherMap = computeLoopDispatcher(AST);

  // Simplify away `SwitchBreakNode`s from `switch`es
  ASTNode *RootNode = simplifySwitchBreakImpl(AST,
                                              AST.getRoot(),
                                              LoopDispatcherMap);
  AST.setRoot(RootNode);

  return RootNode;
}
