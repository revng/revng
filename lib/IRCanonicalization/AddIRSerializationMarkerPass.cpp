//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

/// Pass that wraps Instructions in LLVM IR that must be serialized in
/// special marker calls.

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/Support/Mangling.h"
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

#include "MarkAnalysis.h"
#include "MarkForSerializationFlags.h"

Logger<> MarkLog{ "mark-serialization" };

struct AddIRSerializationMarkersPass : public llvm::FunctionPass {
public:
  static char ID;

  AddIRSerializationMarkersPass() : llvm::FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool runOnFunction(llvm::Function &F) override;
};

bool AddIRSerializationMarkersPass::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  // If the `-single-decompilation` option was passed from command line, skip
  // decompilation for all the functions that are not the selected one.
  if (not TargetFunction.empty())
    if (not F.getName().equals(TargetFunction.c_str()))
      return false;

  // Mark instructions for serialization, and write the results in ToSerialize
  SerializationMap ToSerialize = {};
  MarkAnalysis::Analysis Mark(F, ToSerialize);
  Mark.initialize();
  Mark.run();

  llvm::Module *M = F.getParent();
  llvm::IRBuilder<> Builder(M->getContext());
  bool Changed = false;
  for (const auto &[I, Flag] : ToSerialize) {
    auto *IType = I->getType();

    // We cannot wrap void-typed things into wrappers.
    // We'll have to handle them in another way in the decompilation pipeline
    if (IType->isVoidTy())
      continue;

    if (bool(Flag)) {

      auto *MarkerF = getSerializationMarker(*M, IType);

      // Insert a call to the SCEV barrier right after I. For now the call to
      // barrier has an undef argument, that will be fixed later.
      Builder.SetInsertPoint(I->getParent(), std::next(I->getIterator()));

      // The first argument for the call for now is undef. We'll fix it up
      // later on.
      auto *Undef = llvm::UndefValue::get(IType);

      // The second arg operand needs to be true if the serialization is
      // required because of side effects.
      auto *BoolType = MarkerF->getArg(1)->getType();
      llvm::Constant *MarkSideEffects = nullptr;
      if (Flag.isSet(SerializationReason::HasSideEffects)
          or Flag.isSet(SerializationReason::HasInterferingSideEffects)) {
        MarkSideEffects = llvm::ConstantInt::getAllOnesValue(BoolType);
      } else {
        MarkSideEffects = llvm::ConstantInt::getNullValue(BoolType);
      }

      auto *Call = Builder.CreateCall(MarkerF, { Undef, MarkSideEffects });

      // Replace all uses of I with the new call.
      I->replaceAllUsesWith(Call);

      // Now Fix the call to use I as argument.
      Call->setArgOperand(0, I);

      Changed = true;
    }
  }
  return true;
}

char AddIRSerializationMarkersPass::ID = 0;

using Register = llvm::RegisterPass<AddIRSerializationMarkersPass>;
static Register X("add-ir-serialization-markers",
                  "Pass that adds serialization markers to the IR",
                  false,
                  false);
