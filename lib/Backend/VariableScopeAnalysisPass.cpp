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

#include "revng/ADT/RecursiveCoroutine-coroutine.h"
#include "revng/Support/FunctionTags.h"

#include "revng-c/Backend/VariableScopeAnalysisPass.h"
#include "revng-c/RestructureCFGPass/LoadGHAST.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

using ValuePtrSet = VariableScopeAnalysisPass::ValuePtrSet;

using llvm::BasicBlock;
using llvm::Function;
using llvm::Instruction;
using llvm::User;

using llvm::any_of;
using llvm::cast;

char VariableScopeAnalysisPass::ID = 0;

using Register = llvm::RegisterPass<VariableScopeAnalysisPass>;
static Register X("collect-local-vars",
                  "Collect information about which variables must be "
                  "declared in the outermost scope",
                  false,
                  false);

using VSA = VariableScopeAnalysisPass;

void VSA::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<LoadGHASTWrapperPass>();
  AU.setPreservesAll();
}

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

/// Check if the GHAST has loop dispatchers, which indicate the need for
/// a loop state variable to be declared.
bool hasLoopDispatchers(const ASTTree &GHAST) {
  return needsLoopVar(GHAST.getRoot());
}

/// Fill \a VarsToDeclare with all the llvm::Values for
/// which we need a top-level variable declaration.
ValuePtrSet collectLocalVariables(const Function &F) {
  ValuePtrSet VarsToDeclare;

  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {

      // Only marked instructions have a local variables associated
      auto *Call = llvm::dyn_cast<llvm::CallInst>(&I);
      if (not Call)
        continue;

      llvm::Function *CalledFunction = Call->getCalledFunction();
      if (not CalledFunction)
        continue;
      if (not FunctionTags::AssignmentMarker.isTagOf(CalledFunction))
        continue;

      // If an instruction's uses are all inside the same basic block, we are
      // sure that it can be declared ALAP.
      auto HasDifferentParent = [&I](const User *U) {
        const Instruction *UserInst = cast<const Instruction>(U);
        return UserInst->getParent() != I.getParent();
      };
      bool HasUsersOutsideBB = any_of(I.users(), HasDifferentParent);

      // TODO: we can further refine this logic introducing the concept of
      // scopes and associating variable declarations to a scope.
      // For now, we decided to declare all variables that have at least one
      // use outside of their basic block right at the start of the function.
      if (HasUsersOutsideBB)
        VarsToDeclare.insert(&I);
    }
  }

  return VarsToDeclare;
}

bool VariableScopeAnalysisPass::runOnFunction(Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  // If the `-single-decompilation` option was passed from command line, skip
  // decompilation for all the functions that are not the selected one.
  if (not TargetFunction.empty())
    if (not F.getName().equals(TargetFunction.c_str()))
      return false;

  // Get the Abstract Syntax Tree of the restructured code.
  ASTTree &GHAST = getAnalysis<LoadGHASTWrapperPass>().getGHAST(F);

  TopScopeVariables = collectLocalVariables(F);
  NeedsLoopStateVar = hasLoopDispatchers(GHAST);

  return false;
}
