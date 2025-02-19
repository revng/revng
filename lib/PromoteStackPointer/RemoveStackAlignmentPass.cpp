//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>

#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

#include "revng/PromoteStackPointer/RemoveStackAlignmentPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;
using namespace llvm::PatternMatch;

static bool isBitmaskOperation(Instruction &I) {
  Value *LHS = nullptr;
  ConstantInt *RHS = nullptr;

  if (match(&I, m_And(m_Value(LHS), m_ConstantInt(RHS)))) {
    uint64_t MaskValue = RHS->getZExtValue();
    if (isShiftedMask_64(MaskValue))
      return true;
  }

  return false;
}

static bool isConstantAdd(Instruction &I) {
  Value *LHS = nullptr;
  ConstantInt *RHS = nullptr;

  if (match(&I, m_Add(m_Value(LHS), m_ConstantInt(RHS)))
      or match(&I, m_Sub(m_Value(LHS), m_ConstantInt(RHS)))) {
    return true;
  }

  return false;
}

bool RemoveStackAlignmentPass::runOnModule(Module &Module) {
  if (FunctionTags::Isolated.functions(&Module).empty())
    return false;

  auto *InitFunction = getIRHelper("init_local_sp", Module);
  revng_assert(InitFunction != nullptr);

  bool Result = false;

  SmallVector<User *, 4> Queue;
  for (CallBase *Call : callers(InitFunction))
    llvm::copy(Call->users(), std::back_inserter(Queue));

  while (not Queue.empty()) {
    User *U = Queue.pop_back_val();

    if (auto *I = dyn_cast<Instruction>(U)) {
      if (isBitmaskOperation(*I)) {
        U->replaceAllUsesWith(I->getOperand(0));
        Result = true;
      } else if (isConstantAdd(*I)) {
        llvm::copy(I->users(), std::back_inserter(Queue));
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
