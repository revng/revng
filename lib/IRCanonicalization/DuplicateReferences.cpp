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
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/Support/DecompilationHelpers.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/ModelHelpers.h"

struct DuplicateReferences : public llvm::FunctionPass {
public:
  static char ID;

  DuplicateReferences() : FunctionPass(ID) {}

  // Duplicate reference opcodes that have more than one use. This is needed
  // by the C backend because we cannot emit references is C, so we need to
  // ensure that each reference value has only one use. In this way, we ensure
  // that no reference needs a dedicated variable.
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

using llvm::CallInst;
using llvm::dyn_cast;

bool DuplicateReferences::runOnFunction(llvm::Function &F) {

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

      // Consider only instructions that have more than one use
      if (not I.hasNUsesOrMore(2)) {
        CurInst = NextInst;
        continue;
      }

      if (isCallToTagged(&I, FunctionTags::ModelGEP)
          or isCallToTagged(&I, FunctionTags::ModelGEPRef)) {
        auto *Call = llvm::cast<CallInst>(&I);
        Builder.SetInsertPoint(&I);

        llvm::SmallVector<llvm::User *> Users;
        for (auto *Usr : I.users())
          Users.push_back(Usr);

        for (auto &Usr : Users) {
          // Create a new call
          llvm::SmallVector<llvm::Value *, 16> Args;
          for (auto &Op : Call->arg_operands())
            Args.push_back(Op);

          llvm::Value *NewCall = Builder.CreateCall(Call->getCalledFunction(),
                                                    Args);

          // Substitute the new call
          I.replaceUsesWithIf(NewCall, [&Usr](llvm::Use &U) {
            return U.getUser() == Usr;
          });
        }

        Modified = true;
      }

      CurInst = NextInst;
    }
  }

  return Modified;
}

char DuplicateReferences::ID = 0;

static llvm::RegisterPass<DuplicateReferences> X("duplicate-references",
                                                 "Duplicate reference opcodes "
                                                 "that have more than one use "
                                                 "so that each reference end "
                                                 "up having only one use",
                                                 false,
                                                 false);
