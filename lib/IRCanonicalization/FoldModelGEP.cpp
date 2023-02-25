//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"

#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Model/QualifiedType.h"
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
using model::QualifiedType;

inline llvm::Value *traverseTransparentInstructions(llvm::Value *V) {
  llvm::Value *CurValue = V;

  while (isa<llvm::IntToPtrInst>(CurValue) or isa<llvm::PtrToIntInst>(CurValue))
    CurValue = cast<llvm::Instruction>(CurValue)->getOperand(0);

  return CurValue;
}

static llvm::Value *
getValueToSubstitute(llvm::Instruction &I, const model::Binary &Model) {
  if (auto *Call = llvm::dyn_cast<CallInst>(&I)) {
    revng_log(Log, "--------Call: " << dumpToString(I));

    auto *CalledFunc = Call->getCalledFunction();
    if (not CalledFunc)
      return nullptr;

    if (FunctionTags::ModelGEP.isTagOf(CalledFunc)) {
      // For ModelGEPs, we want to match patterns of the type
      // `ModelGEP(AddressOf())` where the model types recovered by both the
      // ModelGEP and the AddressOf are the same.
      // In this way, we are matching patterns that give birth to expression
      // such as `*&` and `(&base)->` in the decompiled code, which are
      // redundant.

      // First argument is the model type of the base pointer
      llvm::Value *GEPFirstArg = Call->getArgOperand(0);
      QualifiedType GEPBaseType = deserializeFromLLVMString(GEPFirstArg, Model);

      // Second argument is the base pointer
      llvm::Value *SecondArg = Call->getArgOperand(1);
      SecondArg = traverseTransparentInstructions(SecondArg);

      // Skip if the second argument (after traversing casts) is not an
      // AddressOf call
      llvm::CallInst *AddrOfCall = isCallToTagged(SecondArg,
                                                  FunctionTags::AddressOf);
      if (not AddrOfCall)
        return nullptr;

      revng_log(Log, "Second arg is an addressOf ");

      // First argument of the AddressOf is the pointer's base type
      llvm::Value *AddrOfFirstArg = AddrOfCall->getArgOperand(0);
      QualifiedType AddrOfBaseType = deserializeFromLLVMString(AddrOfFirstArg,
                                                               Model);

      // Skip if the ModelGEP is dereferencing the AddressOf with a
      // different type
      if (AddrOfBaseType != GEPBaseType)
        return nullptr;

      revng_log(Log, "Types are the same ");
      revng_log(Log, "Adding " << dumpToString(Call) << " to the map");

      return AddrOfCall->getArgOperand(1);
    }
  }

  return nullptr;
}

bool FoldModelGEP::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

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
  for (auto &BB : llvm::ReversePostOrderTraversal(&F)) {
    auto CurInst = BB->begin();
    while (CurInst != BB->end()) {
      llvm::Instruction &I = *CurInst;
      auto NextInst = std::next(CurInst);

      llvm::Value *ValueToSubstitute = getValueToSubstitute(I, *Model);

      // Prevent merging a ModelGEP with an addressOf that needs a top-scope
      // variable, as this would create
      if (needsTopScopeDeclaration(I)) {
        CurInst = NextInst;
        continue;
      }

      if (ValueToSubstitute) {
        Builder.SetInsertPoint(&I);

        // We should be folding only calls to ModelGEP or AddressOf
        llvm::CallInst *CallToFold = llvm::cast<CallInst>(&I);
        const llvm::Function *CalledFunc = CallToFold->getCalledFunction();
        revng_assert(CalledFunc);

        if (FunctionTags::ModelGEP.isTagOf(CalledFunc)) {

          llvm::SmallVector<llvm::Value *, 8> Args;
          for (auto &Arg : CallToFold->args())
            Args.push_back(Arg);
          Args[1] = ValueToSubstitute;

          auto *ModelGEPRefFunc = getModelGEPRef(*F.getParent(),
                                                 CallToFold->getType(),
                                                 ValueToSubstitute->getType());
          llvm::Value *ModelGEPRef = Builder.CreateCall(ModelGEPRefFunc, Args);

          CallToFold->replaceAllUsesWith(ModelGEPRef);

          ToErase.push_back(CallToFold);
          Modified = true;
        }
      }

      CurInst = NextInst;
    }
  }

  for (auto InstrToErase : ToErase)
    InstrToErase->eraseFromParent();

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
