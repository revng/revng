//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"

#include "revng-c/Support/FunctionTags.h"

struct RemovePointerCasts : public llvm::FunctionPass {
public:
  static char ID;

  RemovePointerCasts() : FunctionPass(ID) {}

  /// Remove ext/trunc casts on integers that are used as pointers.
  ///
  /// E.g., on 32-bit architectures we don't need to extend integers to 64-bits
  /// before using them as pointers, so we want to remove all spurious
  /// `ZExt`/`Trunc` instructions from the IR.
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

using llvm::cast;
using llvm::dyn_cast;
using llvm::Instruction;
using llvm::IntToPtrInst;
using llvm::isa;
using llvm::PtrToIntInst;
using llvm::SExtInst;
using llvm::TruncInst;
using llvm::Value;
using llvm::ZExtInst;

static Instruction *getExtOrTrunc(Value *V) {
  if (isa<ZExtInst>(V) or isa<TruncInst>(V) or isa<SExtInst>(V))
    return cast<Instruction>(V);

  return nullptr;
}

bool RemovePointerCasts::runOnFunction(llvm::Function &F) {

  // Initialize the IR builder to inject instructions
  llvm::LLVMContext &LLVMCtx = F.getContext();
  llvm::IRBuilder<> Builder(LLVMCtx);
  bool Modified = false;

  // TODO: Maybe we should check if the size is compatible with the size of a
  // pointer in the given architecture?

  for (auto &BB : F) {
    auto CurInst = BB.begin();
    while (CurInst != BB.end()) {
      llvm::Instruction &I = *CurInst;
      auto NextInst = std::next(CurInst);

      if (auto *IntToPtr = dyn_cast<IntToPtrInst>(&I)) {
        // Detect Ext/Trunc - IntToPtr patterns
        if (auto *ExtOrTrunc = getExtOrTrunc(IntToPtr->getOperand(0))) {
          Builder.SetInsertPoint(&I);

          // Create a new IntToPtr that skips the cast
          auto *New = Builder.CreateIntToPtr(ExtOrTrunc->getOperand(0),
                                             IntToPtr->getDestTy());

          // Replace the old IntToPtr
          IntToPtr->replaceAllUsesWith(New);

          // Remove unused instructions
          IntToPtr->eraseFromParent();
          if (ExtOrTrunc->hasNUses(0))
            ExtOrTrunc->eraseFromParent();

          Modified = true;
        }
      } else if (auto *Cast = getExtOrTrunc(&I)) {
        // Detect PtrToInt - Ext/Trunc patterns
        if (auto *PtrToInt = dyn_cast<PtrToIntInst>(Cast->getOperand(0))) {
          Builder.SetInsertPoint(&I);

          // Create a new PtrToInt that skips the cast
          auto *New = Builder.CreatePtrToInt(PtrToInt->getOperand(0),
                                             Cast->getType());

          // Replace the cast with the new PtrToInt
          Cast->replaceAllUsesWith(New);

          // Remove unused instructions
          Cast->eraseFromParent();
          if (PtrToInt->hasNUses(0))
            PtrToInt->eraseFromParent();

          Modified = true;
        }
      }

      CurInst = NextInst;
    }
  }

  return Modified;
}

char RemovePointerCasts::ID = 0;

static llvm::RegisterPass<RemovePointerCasts> X("remove-pointer-casts",
                                                "Avoid extending/truncating "
                                                "integers that are used as "
                                                "pointers",
                                                false,
                                                false);
