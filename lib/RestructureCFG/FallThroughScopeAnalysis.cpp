/// \file FallThroughScopeAnalysis.cpp
/// Analysis pass to compute the fallthrough scope
///

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Support/Assert.h"

#include "revng-c/RestructureCFG/ASTNode.h"
#include "revng-c/RestructureCFG/ASTTree.h"
#include "revng-c/RestructureCFG/ExprNode.h"
#include "revng-c/Support/FunctionTags.h"

#include "FallThroughScopeAnalysis.h"

using namespace llvm;

static const model::DynamicFunction &
getDynamicFunction(const model::Binary &Model, StringRef &SymbolName) {
  SymbolName.consume_front("dynamic_");
  auto It = Model.ImportedDynamicFunctions().find(SymbolName.str());
  revng_assert(It != Model.ImportedDynamicFunctions().end());
  return *It;
}

template<typename ModelFunctionOrDynamic>
bool isNoReturn(const ModelFunctionOrDynamic &F) {
  using namespace model::FunctionAttribute;
  return F.Attributes().contains(NoReturn);
}

bool fallsThrough(FallThroughScopeType Element) {
  return Element == FallThroughScopeType::FallThrough;
}

static FallThroughScopeType combineTypes(FallThroughScopeType First,
                                         FallThroughScopeType Second) {

  if (First == Second) {
    return First;
  }

  if (First == FallThroughScopeType::FallThrough
      or Second == FallThroughScopeType::FallThrough) {
    return FallThroughScopeType::FallThrough;
  }

  return FallThroughScopeType::MixedNoFallThrough;
}

static RecursiveCoroutine<FallThroughScopeType>
fallThroughScopeImpl(const model::Binary &Model,
                     ASTNode *Node,
                     FallThroughScopeTypeMap &ResultMap) {
  switch (Node->getKind()) {
  case ASTNode::NK_List: {
    SequenceNode *Seq = llvm::cast<SequenceNode>(Node);

    // Invoke the fallthrough analysis on all the nodes in the sequence node.
    // Even though, after analyzing the sequence node we only use the value of
    // the last node of the sequence, it is important to recursively invoke this
    // routine on all the nodes in the sequence, since in part of the sub-tree
    // other portions of the AST benefiting from this analysis and
    // transformation could exist.
    for (ASTNode *N : Seq->nodes()) {
      FallThroughScopeType NFallThrough = rc_recur
        fallThroughScopeImpl(Model, N, ResultMap);
      ResultMap[N] = NFallThrough;
    }

    // The current sequence node is nofallthrough only if the last node of the
    // sequence node is nofallthrough.
    ASTNode *Last = Seq->getNodeN(Seq->length() - 1);
    rc_return ResultMap.at(Last);
  } break;
  case ASTNode::NK_Scs: {
    ScsNode *Loop = llvm::cast<ScsNode>(Node);

    // The loop node inherits the attribute from the body node of the SCS.
    if (Loop->hasBody()) {
      ASTNode *Body = Loop->getBody();
      FallThroughScopeType BFallThrough = rc_recur
        fallThroughScopeImpl(Model, Body, ResultMap);
      ResultMap[Body] = BFallThrough;
      rc_return BFallThrough;
    } else {
      rc_return FallThroughScopeType::FallThrough;
    }
  } break;
  case ASTNode::NK_If: {
    IfNode *If = llvm::cast<IfNode>(Node);

    // An IfNode is nofallthrough only if both its branches are nofallthrough.
    FallThroughScopeType ThenFallThrough = FallThroughScopeType::FallThrough;
    if (If->hasThen()) {
      ASTNode *Then = If->getThen();
      ThenFallThrough = rc_recur fallThroughScopeImpl(Model, Then, ResultMap);
      ResultMap[Then] = ThenFallThrough;
    }

    FallThroughScopeType ElseFallThrough = FallThroughScopeType::FallThrough;
    if (If->hasElse()) {
      ASTNode *Else = If->getElse();
      ElseFallThrough = rc_recur fallThroughScopeImpl(Model, Else, ResultMap);
      ResultMap[Else] = ElseFallThrough;
    }

    rc_return combineTypes(ThenFallThrough, ElseFallThrough);
  } break;
  case ASTNode::NK_Switch: {
    SwitchNode *Switch = llvm::cast<SwitchNode>(Node);

    // The analysis need to be run even if the results will be decided later on
    // on the basis of other criteria, because we want to compute the
    // `FallThroughScopeType` for the AST subtree beginning at the `SwitchNode`
    // under analysis
    bool FirstIteration = true;
    FallThroughScopeType AllFallThrough;
    for (auto &LabelCasePair : Switch->cases()) {
      ASTNode *Case = LabelCasePair.second;
      FallThroughScopeType CaseFallThrough = rc_recur
        fallThroughScopeImpl(Model, Case, ResultMap);
      ResultMap[Case] = CaseFallThrough;

      // We need to special case the first iteration over the `case`s, so that
      // we initialize the `AllFallThrough` variable with the state that is
      // the lowest over the _lattice_ of `FallThroughScopeType`
      if (FirstIteration) {
        AllFallThrough = CaseFallThrough;
        FirstIteration = false;
      } else {
        AllFallThrough = combineTypes(AllFallThrough, CaseFallThrough);
      }
    }

    // In order to compute the `FallThroughScope` of a `SwitchNode`, we need to
    // take into consideration the following:
    // 1) If we have a standard `SwitchNode`, we can perform the analysis only
    //    if the `default` case is present. This may lead to a suboptimal
    //    result, in cases where the `SwitchNode` does not have the `default`
    //    case, but it however covers all the possible values for the condition
    //    in the enumeration of the cases.
    // 2) If we have a dispatcher `SwitchNode`, we have a guarantee by
    //    construction, that all the `case`s span over the possible values.
    //    Therefore, we can perform the analysis as if the `SwitchNode` had the
    //    `default`.
    if (not Switch->getCondition() or Switch->hasDefault()) {
      rc_return AllFallThrough;
    } else {
      rc_return FallThroughScopeType::FallThrough;
    }
  } break;
  case ASTNode::NK_Code: {
    CodeNode *Code = llvm::cast<CodeNode>(Node);
    llvm::BasicBlock *BB = Code->getBB();
    llvm::Instruction &I = BB->back();

    if (auto *ReturnI = llvm::dyn_cast<ReturnInst>(&I)) {

      // Save the motivation
      ResultMap[Code] = FallThroughScopeType::Return;

      // A return instruction make the current scope `NonLocalCF`
      rc_return FallThroughScopeType::Return;
    } else if (auto *UnreachableI = llvm::dyn_cast<UnreachableInst>(&I)) {

      // In place of an `UnreachableInst`, we should check if we have a call to
      // a `noreturn` function as previous instruction We may not have a
      // previous instruction
      // TODO: confirm the assumption that the call to a `NoReturn` is always
      //       exactly before an `UnreachableInst`, and in case relax this
      //       assumption
      if (Instruction *PrevI = UnreachableI->getPrevNode()) {

        if (const CallInst *Call = getCallToTagged(PrevI,
                                                   FunctionTags::Isolated)) {

          // The called function may be an isolated function. In this case we
          // use the `llvmToModelFunction` helper in order to retrieve the
          // corresponding `model::Function` to check for the `NoReturn`
          // attribute.
          const Function *CalleeFunction = Call->getCalledFunction();
          const model::Function
            *CalleeFunctionModel = llvmToModelFunction(Model, *CalleeFunction);
          if (isNoReturn(*CalleeFunctionModel)) {
            ResultMap[Code] = FallThroughScopeType::CallNoReturn;
            rc_return FallThroughScopeType::CallNoReturn;
          }
        } else if (const CallInst
                     *Call = getCallToTagged(PrevI,
                                             FunctionTags::DynamicFunction)) {

          // The called function may be a dynamic function. In this case, we use
          // the name of the dyamic symbol in order to retrieve the
          // `model::DynamicFunction` and check for the `NoReturn` attribute.
          const Function *CalleeFunction = Call->getCalledFunction();
          llvm::StringRef SymbolName = CalleeFunction->getName()
                                         .drop_front(strlen("dynamic_"));
          const model::DynamicFunction
            &CalleeFunctionModel = getDynamicFunction(Model, SymbolName);
          if (isNoReturn(CalleeFunctionModel)) {
            ResultMap[Code] = FallThroughScopeType::CallNoReturn;
            rc_return FallThroughScopeType::CallNoReturn;
          }
        }
      }
    }

    rc_return FallThroughScopeType::FallThrough;
  } break;
  case ASTNode::NK_Set: {
    rc_return FallThroughScopeType::FallThrough;
  } break;
  case ASTNode::NK_SwitchBreak: {
    rc_return FallThroughScopeType::SwitchBreak;
  } break;
  case ASTNode::NK_Continue: {
    rc_return FallThroughScopeType::Continue;
  }
  case ASTNode::NK_Break: {
    rc_return FallThroughScopeType::LoopBreak;
  } break;
  default:
    revng_abort();
  }

  rc_return FallThroughScopeType::FallThrough;
}

FallThroughScopeTypeMap computeFallThroughScope(const model::Binary &Model,
                                                ASTNode *RootNode) {
  FallThroughScopeTypeMap ResultMap;
  FallThroughScopeType Result = fallThroughScopeImpl(Model,
                                                     RootNode,
                                                     ResultMap);
  ResultMap[RootNode] = Result;
  return ResultMap;
}
