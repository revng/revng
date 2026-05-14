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

static llvm::BasicBlock *findUseDominator(const llvm::DominatorTree &DT,
                                          llvm::Value *Value) {
  llvm::BasicBlock *Dominator = nullptr;
  for (llvm::User *User : Value->users()) {
    llvm::BasicBlock *BB = llvm::cast<llvm::Instruction>(User)->getParent();
    Dominator = Dominator == nullptr ?
                  BB :
                  DT.findNearestCommonDominator(Dominator, BB);
  }
  return Dominator;
}

static llvm::Instruction *findFirstUserInBlock(llvm::Value *Value,
                                               llvm::BasicBlock *BB) {
  llvm::Instruction *FirstUser = nullptr;
  for (llvm::User *User : Value->users()) {
    llvm::Instruction *I = llvm::cast<llvm::Instruction>(User);
    if (I->getParent() == BB) {
      if (FirstUser == nullptr or I->comesBefore(FirstUser))
        FirstUser = I;
    }
  }
  return FirstUser;
}

bool DeferAllocasPass::runOnFunction(llvm::Function &F) {
  llvm::DominatorTree DT;
  DT.recalculate(F);

  using DeferrableAlloca = std::tuple<llvm::AllocaInst *, //
                                      llvm::BasicBlock *,
                                      llvm::BasicBlock::iterator>;

  llvm::SmallVector<DeferrableAlloca> DeferrableAllocas;
  for (llvm::BasicBlock &BB : F) {
    for (llvm::Instruction &I : BB) {
      auto Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I);
      if (not Alloca or hasStackFrameMetadata(Alloca))
        continue;

      llvm::BasicBlock *NewBB = findUseDominator(DT, Alloca);
      if (NewBB != nullptr) {
        llvm::Instruction *FirstUser = findFirstUserInBlock(Alloca, NewBB);

        if (FirstUser == nullptr)
          FirstUser = NewBB->getTerminator();

        llvm::BasicBlock::iterator NewPos = FirstUser->getIterator();
        if (NewBB != &BB or NewPos != std::next(Alloca->getIterator()))
          DeferrableAllocas.emplace_back(Alloca, NewBB, NewPos);
      }
    }
  }

  for (auto [Alloca, NewBB, NewPos] : DeferrableAllocas)
    Alloca->moveBefore(*NewBB, NewPos);

  return not DeferrableAllocas.empty();
}

static llvm::RegisterPass<DeferAllocasPass> X("defer-allocas",
                                              "Transformation pass which moves "
                                              "alloca instructions to the "
                                              "basic block dominating all uses "
                                              "of the alloca value.");

} // namespace
