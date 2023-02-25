//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#include "revng/ADT/SmallMap.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using llvm::AllocaInst;
using llvm::AnalysisUsage;
using llvm::BasicBlock;
using llvm::Function;
using llvm::FunctionPass;
using llvm::Instruction;
using llvm::IRBuilder;
using llvm::PHINode;
using llvm::RegisterPass;
using llvm::Value;

static Logger<> Log{ "exit-ssa" };

struct ExitSSAPass : public FunctionPass {
public:
  static char ID;

  ExitSSAPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

bool ExitSSAPass::runOnFunction(Function &F) {

  // Skip non-isolated functions
  if (not FunctionTags::Isolated.isTagOf(&F))
    return false;

  IRBuilder<> Builder(F.getContext());
  Builder.SetInsertPoint(&F.getEntryBlock().front());

  SmallMap<PHINode *, AllocaInst *, 8> PHIToAlloca;
  for (BasicBlock &BB : F)
    for (PHINode &PHI : BB.phis())
      PHIToAlloca[&PHI] = Builder.CreateAlloca(PHI.getType());

  if (PHIToAlloca.empty())
    return false;

  for (auto &[PHI, Alloca] : PHIToAlloca) {
    for (auto &IncomingUse : PHI->incoming_values()) {
      llvm::BasicBlock *BB = PHI->getIncomingBlock(IncomingUse);
      Builder.SetInsertPoint(BB->getTerminator());
      Value *Incoming = IncomingUse.get();

      BasicBlock *IncomingDefBB = &F.getEntryBlock();
      if (auto *I = dyn_cast<Instruction>(Incoming))
        IncomingDefBB = I->getParent();
      revng_assert(IncomingDefBB);

      revng_log(Log, "Incoming: " << dumpToString(Incoming));
      auto *S = Builder.CreateStore(Incoming, PHIToAlloca.at(PHI));
      revng_log(Log, dumpToString(S));
    }
  }

  for (auto &[PHI, Alloca] : PHIToAlloca) {
    Builder.SetInsertPoint(PHI);
    auto *Load = createLoad(Builder, Alloca);
    PHI->replaceAllUsesWith(Load);
    PHI->eraseFromParent();
  }

  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      revng_assert(not llvm::isa<PHINode>(I));

  return not PHIToAlloca.empty();
}

char ExitSSAPass::ID = 0;

static RegisterPass<ExitSSAPass> X("exit-ssa",
                                   "Transformation pass that exits from Static "
                                   "Single Assignment form, promoting PHINodes "
                                   "to sets of Allocas, Load and Stores",
                                   false,
                                   false);
