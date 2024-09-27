//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"

#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/Support/DecompilationHelpers.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/ModelHelpers.h"

static Logger<> Log{ "fold-model-gep" };

struct FoldModelGEP : public llvm::FunctionPass {
public:
  static char ID;

  FoldModelGEP() : FunctionPass(ID) {}

  /// Fold all ModelGEP(type1, AddressOf(type2, %value)) to %value when type1
  /// and type2 are the same
  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<LoadModelWrapperPass>();
    AU.setPreservesCFG();
  }
};

using llvm::CallInst;
using llvm::dyn_cast;
using llvm::StringRef;

inline llvm::Value *traverseTransparentInstructions(llvm::Value *V) {
  llvm::Value *CurValue = V;

  while (isa<llvm::IntToPtrInst>(CurValue) or isa<llvm::PtrToIntInst>(CurValue)
         or isa<llvm::BitCastInst>(CurValue) or isa<llvm::FreezeInst>(CurValue))
    CurValue = cast<llvm::Instruction>(CurValue)->getOperand(0);

  return CurValue;
}

static llvm::Value *getValueToSubstitute(llvm::Instruction &I,
                                         const model::Binary &Model) {
  if (auto *Call = getCallToTagged(&I, FunctionTags::ModelGEP)) {
    revng_log(Log, "--------Call: " << dumpToString(I));

    // For ModelGEPs, we want to match patterns of the type
    // `ModelGEP(AddressOf())` where the model types recovered by both the
    // ModelGEP and the AddressOf are the same.
    // In this way, we are matching patterns that give birth to expression
    // such as `*&` and `(&base)->` in the decompiled code, which are
    // redundant.

    // First argument is the model type of the base pointer
    llvm::Value *GEPFirstArg = Call->getArgOperand(0);
    auto GEPBaseType = fromLLVMString(GEPFirstArg, Model);

    // Second argument is the base pointer
    llvm::Value *SecondArg = Call->getArgOperand(1);
    SecondArg = traverseTransparentInstructions(SecondArg);

    // Skip if the ModelGEP is doing an array access. If it's doing an array
    // access, we'd lose that by folding, so we only traverse if it's constant
    // and it's zero.
    llvm::Value *ThirdArg = Call->getArgOperand(2);
    auto *ConstantArrayIndex = dyn_cast<llvm::ConstantInt>(ThirdArg);
    bool HasInitialArrayAccess = not ConstantArrayIndex
                                 or not ConstantArrayIndex->isZero();
    if (HasInitialArrayAccess)
      return nullptr;

    // Skip if the second argument (after traversing casts) is not an
    // AddressOf call
    llvm::CallInst *AddrOfCall = getCallToTagged(SecondArg,
                                                 FunctionTags::AddressOf);
    if (not AddrOfCall)
      return nullptr;

    revng_log(Log, "Second arg is an addressOf ");

    // First argument of the AddressOf is the pointer's base type
    llvm::Value *AddrOfFirstArg = AddrOfCall->getArgOperand(0);
    auto AddrOfBaseType = fromLLVMString(AddrOfFirstArg, Model);

    // Skip if the ModelGEP is dereferencing the AddressOf with a
    // different type
    if (AddrOfBaseType != GEPBaseType)
      return nullptr;

    revng_log(Log, "Types are the same ");
    revng_log(Log, "Adding " << dumpToString(Call) << " to the map");

    return AddrOfCall->getArgOperand(1);
  }

  return nullptr;
}

bool FoldModelGEP::runOnFunction(llvm::Function &F) {

  // Get the model
  const auto
    &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel().get();

  // Initialize the IR builder to inject functions
  llvm::LLVMContext &LLVMCtx = F.getContext();
  llvm::IRBuilder<> Builder(LLVMCtx);
  bool Modified = false;

  revng_log(Log, "=========Function: " << F.getName());

  llvm::SmallVector<llvm::Instruction *, 32> ToErase;

  // Collect ModelGEPs
  for (auto *BB : llvm::ReversePostOrderTraversal(&F)) {
    for (auto &I : llvm::make_early_inc_range(*BB)) {

      if (llvm::Value *ValueToSubstitute = getValueToSubstitute(I, *Model)) {
        auto *CallToFold = cast<CallInst>(&I);
        revng_assert(isCallToTagged(CallToFold, FunctionTags::ModelGEP));
        Builder.SetInsertPoint(CallToFold);

        llvm::SmallVector<llvm::Value *, 8> Args;
        for (auto &Group : llvm::enumerate(CallToFold->args())) {
          llvm::Value *Arg = Group.value();
          // We just ignore the argument representing the array index for the
          // ModelGEPRef.
          if (Group.index() == 2) {
            revng_assert(isa<llvm::ConstantInt>(Arg)
                         and cast<llvm::ConstantInt>(Arg)->isZero());
            continue;
          }
          Args.push_back(Arg);
        }
        Args[1] = ValueToSubstitute;

        auto *ModelGEPRefFunc = getModelGEPRef(*F.getParent(),
                                               CallToFold->getType(),
                                               ValueToSubstitute->getType());
        llvm::Value *ModelGEPRef = Builder.CreateCall(ModelGEPRefFunc, Args);

        CallToFold->replaceAllUsesWith(ModelGEPRef);
        CallToFold->eraseFromParent();

        Modified = true;
      }
    }
  }

  return Modified;
}

char FoldModelGEP::ID = 0;

static llvm::RegisterPass<FoldModelGEP> X("fold-model-gep",
                                          "Folds ModelGEPs/AddressOf "
                                          "patterns that might produce *&, &* "
                                          "or (&...)-> expressions "
                                          "in the decompiled code.",
                                          false,
                                          false);
