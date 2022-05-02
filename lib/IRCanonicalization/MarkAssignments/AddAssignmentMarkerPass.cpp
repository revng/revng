//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

/// Pass that detects Instructions in a Functions for which we have to generate
/// a variable assignment when decompiling to C, and wraps them in special
/// marker calls.

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

#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/Mangling.h"

#include "MarkAssignments.h"

struct AddAssignmentMarkersPass : public llvm::FunctionPass {
public:
  static char ID;

  AddAssignmentMarkersPass() : llvm::FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool runOnFunction(llvm::Function &F) override;
};

bool AddAssignmentMarkersPass::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  MarkAssignments::AssignmentMap
    Assignments = MarkAssignments::selectAssignments(F);

  llvm::Module *M = F.getParent();
  llvm::IRBuilder<> Builder(M->getContext());
  bool Changed = false;
  for (auto &[I, Flag] : Assignments) {
    auto *IType = I->getType();

    // We cannot wrap void-typed things into wrappers.
    // We'll have to handle them in another way in the decompilation pipeline
    if (IType->isVoidTy())
      continue;

    if (bool(Flag)) {

      auto *MarkerF = getAssignmentMarker(*M, IType);

      // Insert a call to the SCEV barrier right after I. For now the call to
      // barrier has an undef argument, that will be fixed later.
      Builder.SetInsertPoint(I->getParent(), std::next(I->getIterator()));

      // The first argument for the call for now is undef. We'll fix it up
      // later on.
      auto *Undef = llvm::UndefValue::get(IType);

      // The second arg operand needs to be true if the assignment is
      // required because of side effects.
      auto *BoolType = MarkerF->getArg(1)->getType();
      auto *MarkSideEffects = Flag.hasSideEffects() ?
                                llvm::ConstantInt::getAllOnesValue(BoolType) :
                                llvm::ConstantInt::getNullValue(BoolType);

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

char AddAssignmentMarkersPass::ID = 0;

using Register = llvm::RegisterPass<AddAssignmentMarkersPass>;
static Register X("add-assignment-markers",
                  "Pass that adds assignment markers to the IR",
                  false,
                  false);
