//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

#include "revng/LocalVariables/LocalVariableHelpers.h"

namespace {

struct DeferAllocasPass : public llvm::FunctionPass {
public:
  static char ID;

  DeferAllocasPass() : FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
};

char DeferAllocasPass::ID = 0;

bool DeferAllocasPass::runOnFunction(llvm::Function &F) {
  llvm::DominatorTree DT;
  DT.recalculate(F);

  bool Modified = false;
  for (llvm::BasicBlock &BB : F) {
    // Iterate manually to avoid invalidating the iterator if and when the
    // alloca instruction is moved out of the basic block.
    for (llvm::Instruction &I : llvm::make_early_inc_range(BB)) {
      auto Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I);

      if (not Alloca or hasStackFrameMetadata(Alloca))
        continue;

      llvm::BasicBlock *NewBB = nullptr;
      for (llvm::User *User : Alloca->users()) {
        llvm::BasicBlock *UserBB = //
          llvm::cast<llvm::Instruction>(User)->getParent();

        NewBB = NewBB == nullptr ? UserBB :
                                   DT.findNearestCommonDominator(NewBB, UserBB);
      }

      if (NewBB != nullptr and NewBB != &BB) {
        Alloca->moveBefore(*NewBB, NewBB->begin());
        Modified = true;
      }
    }
  }

  return Modified;
}

static llvm::RegisterPass<DeferAllocasPass> X("defer-allocas",
                                              "Transformation pass which moves "
                                              "alloca instructions to the "
                                              "basic block dominating all uses "
                                              "of the alloca value.");

} // namespace
