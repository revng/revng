//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {

/// `inttoptr` already does an implicit zero-extension when its operand is
/// narrower than the target pointer, and `ptrtoint` already truncates when the
/// target integer is narrower than the pointer.
/// This pass exploits this fact turning:
///
///   %w = zext iN %x to iM
///   %p = inttoptr iM %w to ptr
///   %w = ptrtoint ptr %p to iM
///   %t = trunc iM %w to iN
///
/// into:
///
///   %p = inttoptr iN %x to ptr
///   %t = ptrtoint ptr %p to iN
///
/// \note instcombine reverts these changes, so you might want to run this
//        close to the leaves of your pipeline.
struct DropRedundantPtrIntCasts : public FunctionPass {
  static char ID;

  DropRedundantPtrIntCasts() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool runOnFunction(Function &F) override {
    bool Changed = false;
    SmallVector<Instruction *, 16> ToErase;

    for (auto &I : llvm::instructions(F)) {
      if (auto *IntToPtr = dyn_cast<IntToPtrInst>(&I)) {
        if (auto *ZeroExtend = dyn_cast<ZExtInst>(IntToPtr->getOperand(0))) {
          IRBuilder<> B(IntToPtr);
          auto *New = B.CreateIntToPtr(ZeroExtend->getOperand(0),
                                       IntToPtr->getType());
          if (auto *NewI = dyn_cast<Instruction>(New))
            NewI->copyMetadata(*IntToPtr);
          IntToPtr->replaceAllUsesWith(New);
          ToErase.push_back(IntToPtr);
          Changed = true;
        }
      } else if (auto *TR = dyn_cast<TruncInst>(&I)) {
        if (auto *PtrToInt = dyn_cast<PtrToIntInst>(TR->getOperand(0))) {
          IRBuilder<> B(TR);
          auto *New = B.CreatePtrToInt(PtrToInt->getPointerOperand(),
                                       TR->getType());
          if (auto *NewI = dyn_cast<Instruction>(New))
            NewI->copyMetadata(*TR);
          TR->replaceAllUsesWith(New);
          ToErase.push_back(TR);
          Changed = true;
        }
      }
    }

    for (auto *I : ToErase)
      I->eraseFromParent();

    return Changed;
  }
};

} // namespace

char DropRedundantPtrIntCasts::ID = 0;

using Register = RegisterPass<DropRedundantPtrIntCasts>;
static Register X("drop-redundant-ptrint-casts", "", false, false);
