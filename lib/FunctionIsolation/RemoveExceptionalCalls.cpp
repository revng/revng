/// \file RemoveExceptionalCalls.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "revng/FunctionIsolation/RemoveExceptionalCalls.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

char RemoveExceptionalCalls::ID = 0;

using Register = RegisterPass<RemoveExceptionalCalls>;
static Register X("remove-exceptional-functions",
                  "Remove Exceptional Functions");

bool RemoveExceptionalCalls::runOnModule(llvm::Module &M) {
  LLVMContext &C = M.getContext();

  // Collect all calls to exceptional functions in lifted functions
  std::set<CallBase *> ToErase;
  for (Function &F : FunctionTags::Exceptional.functions(&M))
    for (CallBase *Call : callers(&F))
      if (FunctionTags::Isolated.isTagOf(Call->getParent()->getParent()))
        ToErase.insert(Call);

  std::set<Function *> ToCleanup;
  for (CallBase *Call : ToErase) {
    BasicBlock *BB = Call->getParent();

    // Register function for cleanup
    ToCleanup.insert(BB->getParent());

    // Split containing basic block
    BB->splitBasicBlock(Call);

    // TODO: investigate whether it's safe to use this after the termination
    //       was dropped.
    DILocation *DI = BB->getTerminator()->getDebugLoc();

    // Drop terminator of the old basic block
    eraseFromParent(BB->getTerminator());

    // Terminate with an unreachable
    auto Unreachable = new UnreachableInst(C, BB);
    Unreachable->setDebugLoc(DI);

    // Drop function call
    auto *Undef = UndefValue::get(Call->getType());
    Call->replaceAllUsesWith(Undef);
    eraseFromParent(Call);
  }

  // Garbage collect dead blocks
  for (Function *F : ToCleanup)
    EliminateUnreachableBlocks(*F, nullptr, false);

  return true;
}
