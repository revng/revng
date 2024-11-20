//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

#include "revng/ABI/ModelHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/FunctionTags.h"

using namespace llvm;

struct CleanupIRPass : public ModulePass {
public:
  static char ID;

  CleanupIRPass() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
  }

private:
  class Impl {

  private:
    Module &M;
    LLVMContext &Context;
    const model::Binary &Model;

  public:
    Impl(Module &TheModule, const model::Binary &TheModel) :
      M(TheModule), Context(M.getContext()), Model(TheModel) {}

    bool run();

  private:
    bool replaceInstructions(Function &F);
  };
};

bool CleanupIRPass::Impl::replaceInstructions(Function &F) {

  bool Changed = false;

  for (Instruction &I : llvm::make_early_inc_range(llvm::instructions(F))) {

    if (auto *Call = getCallToTagged(&I, FunctionTags::AddressOf)) {
      Call->replaceAllUsesWith(Call->getArgOperand(1));
      Call->eraseFromParent();
      Changed = true;
    } else if (auto *Call = getCallToTagged(&I, FunctionTags::StringLiteral)) {
      auto *PtrToString = cast<Constant>(Call->getArgOperand(0));
      if (Call->getType()->isIntegerTy()) {
        Call->replaceAllUsesWith(ConstantExpr::getPtrToInt(PtrToString,
                                                           Call->getType()));
      } else if (Call->getType()->isPointerTy()) {
        Call->replaceAllUsesWith(PtrToString);
      } else {
        Call->dump();
        Call->getFunction()->dump();
        revng_abort();
      }

      Call->eraseFromParent();
      Changed = true;
    }
  }

  for (Instruction &I : llvm::make_early_inc_range(llvm::instructions(F))) {
    if (auto *Call = getCallToTagged(&I,
                                     FunctionTags::AllocatesLocalVariable)) {
      IRBuilder<> Builder(Context);
      Builder.SetInsertPointPastAllocas(Call->getFunction());
      Value *AllocatedSize = nullptr;
      if (auto *Callee = getCalledFunction(Call);
          Callee and Callee->getName().startswith("revng_stack_frame")) {
        AllocatedSize = Call->getArgOperand(0);
      } else {
        model::UpcastableType
          AllocatedType = fromLLVMString(Call->getArgOperand(0), Model);
        AllocatedSize = ConstantInt::get(Context,
                                         APInt(/*NumBits*/ 64,
                                               AllocatedType->size().value()));
      }
      auto *Int8Type = IntegerType::getInt8Ty(Context);
      auto *Alloca = Builder.CreateAlloca(Int8Type,
                                          /* ArraySize */ AllocatedSize);

      // Some uses of the Call can be replaced directly with GEPs in the Alloca.
      for (Use &U : Call->uses()) {
        User *TheUser = U.getUser();
        // If a use is an add, whose result is casted to pointer, then we can
        // just replace all the uses of the IntToPtr with a GEP in the Alloca.
        if (auto *BinOp = dyn_cast<BinaryOperator>(TheUser);
            BinOp and BinOp->getOpcode() == Instruction::Add) {
          for (Use &BinOpUse : BinOp->uses()) {
            User *BinOpUser = BinOpUse.getUser();
            if (auto *IntToPtr = dyn_cast<IntToPtrInst>(BinOpUser)) {
              unsigned OtherOperandIndex = U.getOperandNo() ? 0 : 1;
              Value *OtherOperand = BinOp->getOperand(OtherOperandIndex);
              Builder.SetInsertPoint(IntToPtr);
              auto *GEP = Builder.CreateGEP(Int8Type, Alloca, { OtherOperand });
              IntToPtr->replaceAllUsesWith(GEP);
            }
          }
        }
        // If a use is an IntToPtr, we can just use the Alloca instead
        if (auto *IntToPtr = dyn_cast<IntToPtrInst>(TheUser)) {
          IntToPtr->replaceAllUsesWith(Alloca);
        }
      }

      // If there are other uses left, replace them more cautiously.
      if (Call->getNumUses()) {
        Builder.SetInsertPoint(Call);
        auto *PtrToInt = Builder.CreatePtrToInt(Alloca, Call->getType());
        Call->replaceAllUsesWith(PtrToInt);
      }

      Call->eraseFromParent();
      Changed = true;
      continue;
    }
  }

  return Changed;
}

bool CleanupIRPass::Impl::run() {
  bool Changed = false;

  // First, look at the body of of each isolated function, and for each call to
  // a custom opcode replace it with something LLVM-native with equivalent
  // semantics.
  for (Function &F : FunctionTags::Isolated.functions(&M))
    Changed |= replaceInstructions(F);

  return Changed;
}

bool CleanupIRPass::runOnModule(Module &TheModule) {

  auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const model::Binary &Model = *ModelWrapper.getReadOnlyModel();

  return Impl(TheModule, Model).run();
}

char CleanupIRPass::ID = 0;

using Reg = RegisterPass<CleanupIRPass>;
static Reg X("cleanup-ir", "CleanupIRPass");
