//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"

#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Support/Assert.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/Support/DecompilationHelpers.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/ModelHelpers.h"

struct RemoveRefDeref : public llvm::FunctionPass {
public:
  static char ID;

  RemoveRefDeref() : FunctionPass(ID) {}

  // Remove `ModelGEPRef` functions with no arguments
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

using llvm::CallInst;
using llvm::dyn_cast;

static llvm::CallInst *isRefDeref(llvm::Value *V) {
  if (auto *Call = getCallToTagged(V, FunctionTags::ModelGEPRef))
    if (Call->arg_size() == 2)
      return Call;

  return nullptr;
}

bool RemoveRefDeref::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  // Initialize the IR builder to inject functions
  llvm::LLVMContext &LLVMCtx = F.getContext();
  llvm::IRBuilder<> Builder(LLVMCtx);
  bool Modified = false;

  for (auto &BB : F) {
    auto CurInst = BB.begin();
    while (CurInst != BB.end()) {
      llvm::Instruction &I = *CurInst;
      auto NextInst = std::next(CurInst);

      if (auto *Call = isRefDeref(&I)) {
        Builder.SetInsertPoint(&I);

        auto *BasePtrArg = Call->getArgOperand(1);
        llvm::Type *BasePtrType = BasePtrArg->getType();
        llvm::Type *CallRetType = Call->getType();

        if (BasePtrType != CallRetType) {
          // We consider pointers and integers on the LLVM IR interchangable,
          // since it's the model who decides whether a values is a pointer or
          // not
          if (BasePtrType->isPointerTy() and CallRetType->isIntegerTy()) {
            BasePtrArg = Builder.CreatePtrToInt(BasePtrArg, CallRetType);

          } else if (BasePtrType->isIntegerTy()
                     and CallRetType->isPointerTy()) {
            BasePtrArg = Builder.CreateIntToPtr(BasePtrArg, CallRetType);

          } else if (BasePtrType->isIntegerTy()
                     and CallRetType->isIntegerTy()) {
            // A ModelGEPRef with no arguments and model type T takes a
            // reference of type T as argument and returns a reference of type
            // T.
            // Since by constructions references have the size of the
            // referenced value in LLVM IR, when simplifying a ModelGEPRef
            // with no arguments we should always be substituting a value
            // with size sizeof(T) with a value of the same size.
            //
            // The only possible exception is the use of booleans in the IR
            // (i1). In that case, the smallest model type we can assign to
            // them is a byte, and they are loaded and stored as bytes, but
            // when we use them in the IR the actual size of the type is 1.
            auto BaseSize = BasePtrType->getScalarSizeInBits();
            auto RetSize = CallRetType->getScalarSizeInBits();
            revng_assert((RetSize == 8 and BaseSize == 1)
                         or (RetSize == 1 and BaseSize == 8));

            BasePtrArg = Builder.CreateZExtOrTrunc(BasePtrArg, Call->getType());
          } else {
            revng_abort("Incompatible types between ModelGEPRef and its base.");
          }
        }

        Call->replaceAllUsesWith(BasePtrArg);

        Call->eraseFromParent();
        Modified = true;
      }

      CurInst = NextInst;
    }
  }

  return Modified;
}

char RemoveRefDeref::ID = 0;

static llvm::RegisterPass<RemoveRefDeref> X("remove-ref-deref",
                                            "Remove all dereferences of "
                                            "reference values",
                                            false,
                                            false);
