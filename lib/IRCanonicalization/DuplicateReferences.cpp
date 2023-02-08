//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
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

bool DuplicateReferences::runOnFunction(llvm::Function &F) {

  // Skip non-isolated functions
  auto FTags = FunctionTags::TagsSet::from(&F);
  if (not FTags.contains(FunctionTags::Isolated))
    return false;

  // Initialize the IR builder to inject functions
  llvm::IRBuilder<> Builder(F.getContext());
  bool Modified = false;

  for (llvm::BasicBlock *BB : llvm::post_order(&F)) {
    for (llvm::Instruction &I : llvm::reverse(*BB)) {

      // Skip all instructions with less than 2 uses because there's nothing to
      // duplicate.
      if (not I.hasNUsesOrMore(2))
        continue;

      if (isCallToTagged(&I, FunctionTags::ModelGEP)
          or isCallToTagged(&I, FunctionTags::ModelGEPRef)
          or isCallToTagged(&I, FunctionTags::SegmentRef)) {

        auto *Call = llvm::cast<llvm::CallInst>(&I);
        auto *CalledFunction = Call->getCalledFunction();

        // Insert the duplicated calls right after the original call.
        Builder.SetInsertPoint(Call->getNextNode());

        // Insert a duplicated call for each use except the first, that can keep
        // using the old one, and update the use to use the duplicate.
        for (llvm::Use &U :
             llvm::make_early_inc_range(llvm::drop_begin(I.uses()))) {
          auto Args = llvm::SmallVector<llvm::Value *>{ Call->arg_operands() };
          U.set(Builder.CreateCall(CalledFunction, Args));
          Modified = true;
        }
      }
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
