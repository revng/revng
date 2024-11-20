//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#include "revng/PromoteStackPointer/RemoveStackAlignmentPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

static bool isMask(Value *V, unsigned MaxMaskedBits) {
  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    APInt Value = CI->getValue();
    Value.flipAllBits();
    return (Value.isMask() and Value.countTrailingOnes() <= MaxMaskedBits);
  }

  return false;
}

bool RemoveStackAlignmentPass::runOnModule(Module &Module) {
  if (FunctionTags::Isolated.functions(&Module).empty())
    return false;

  auto *InitFunction = Module.getFunction("_init_local_sp");
  revng_assert(InitFunction != nullptr);

  bool Result = false;
  for (CallBase *Call : callers(InitFunction)) {
    for (User *U : Call->users()) {
      if (auto *I = dyn_cast<BinaryOperator>(U)) {
        if (I->getOpcode() == Instruction::And
            and isMask(I->getOperand(1), 12)) {
          U->replaceAllUsesWith(I->getOperand(0));
          Result = true;
        }
      }
    }
  }

  return Result;
}

void RemoveStackAlignmentPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
}

char RemoveStackAlignmentPass::ID = 0;

using Register = RegisterPass<RemoveStackAlignmentPass>;
static Register R("remove-stack-alignment", "Remove Stack Alignment Pass");
