//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/Backend/VariableScopeAnalysis.h"
#include "revng-c/RestructureCFG/ASTNode.h"
#include "revng-c/RestructureCFG/ASTTree.h"
#include "revng-c/Support/FunctionTags.h"

using ValuePtrSet = llvm::SmallPtrSet<const llvm::Instruction *, 32>;

using llvm::BasicBlock;
using llvm::CallInst;
using llvm::Function;
using llvm::Instruction;
using llvm::User;

using llvm::any_of;
using llvm::cast;

/// Visit the node and all its children recursively, checking if a loop
/// variable is needed.
// TODO: This could be precomputed and attached to the SCS node in the GHAST.
static RecursiveCoroutine<bool> needsLoopVar(ASTNode *N) {
  if (N == nullptr)
    rc_return false;

  auto Kind = N->getKind();
  switch (Kind) {

  case ASTNode::NodeKind::NK_Break:
  case ASTNode::NodeKind::NK_SwitchBreak:
  case ASTNode::NodeKind::NK_Continue:
  case ASTNode::NodeKind::NK_Code:
    rc_return false;
    break;

  case ASTNode::NodeKind::NK_If: {
    IfNode *If = cast<IfNode>(N);

    if (nullptr != If->getThen())
      if (rc_recur needsLoopVar(If->getThen()))
        rc_return true;

    if (If->hasElse())
      if (rc_recur needsLoopVar(If->getElse()))
        rc_return true;

    rc_return false;
  } break;

  case ASTNode::NodeKind::NK_Scs: {
    ScsNode *LoopBody = cast<ScsNode>(N);
    rc_return rc_recur needsLoopVar(LoopBody->getBody());
  } break;

  case ASTNode::NodeKind::NK_List: {
    SequenceNode *Seq = cast<SequenceNode>(N);
    for (ASTNode *Child : Seq->nodes())
      if (rc_recur needsLoopVar(Child))
        rc_return true;

    rc_return false;
  } break;

  case ASTNode::NodeKind::NK_Switch: {
    SwitchNode *Switch = cast<SwitchNode>(N);
    llvm::Value *SwitchVar = Switch->getCondition();

    if (not SwitchVar)
      rc_return true;

    for (const auto &[Labels, CaseNode] : Switch->cases())
      if (rc_recur needsLoopVar(CaseNode))
        rc_return true;

    if (auto *Default = Switch->getDefault())
      if (rc_recur needsLoopVar(Default))
        rc_return true;

    rc_return false;
  } break;

  case ASTNode::NodeKind::NK_Set: {
    rc_return true;
  } break;
  }
}

bool hasLoopDispatchers(const ASTTree &GHAST) {
  return needsLoopVar(GHAST.getRoot());
}

ValuePtrSet collectTopScopeVariables(const Function &F) {
  ValuePtrSet TopScopeVars;

  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {

      // We always want to put the stack frame among the top-scope variables,
      // since it is is logical for it to appear at the top of the function even
      // if it is used only in a later scope.
      if (isCallTo(&I, "revng_stack_frame") or needsTopScopeDeclaration(I))
        TopScopeVars.insert(&I);
    }
  }

  return TopScopeVars;
}
