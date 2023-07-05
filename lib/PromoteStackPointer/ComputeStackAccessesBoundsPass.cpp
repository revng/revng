//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Analysis/LazyValueInfo.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Instructions.h"

#include "revng/Model/IRHelpers.h"

#include "revng-c/PromoteStackPointer/ComputeStackAccessesBoundsPass.h"
#include "revng-c/PromoteStackPointer/InstrumentStackAccessesPass.h"
#include "revng-c/Support/FunctionTags.h"

using namespace llvm;

bool ComputeStackAccessesBoundsPass::runOnModule(Module &M) {
  for (Function &StackOffsetFunction :
       FunctionTags::StackOffsetMarker.functions(&M)) {
    auto *FunctionType = StackOffsetFunction.getFunctionType();
    auto *DifferenceType = cast<IntegerType>(FunctionType->getParamType(1));
    for (CallBase *Call : callers(&StackOffsetFunction)) {
      Function &F = *Call->getParent()->getParent();
      LazyValueInfo &LVI = getAnalysis<LazyValueInfoWrapperPass>(F).getLVI();
      auto *Undef = UndefValue::get(DifferenceType);

      // Identify lower bound of the lower bound
      const auto &LowerBoundRange = LVI.getConstantRange(Call->getArgOperand(1),
                                                         cast<CallInst>(Call));
      Value *LowerBound = nullptr;
      if (not LowerBoundRange.isFullSet()) {
        LowerBound = ConstantInt::get(DifferenceType,
                                      LowerBoundRange.getLower());
      } else {
        LowerBound = Undef;
      }
      Call->setArgOperand(1, LowerBound);

      // Identify upper bound of the upper bound
      const auto &UpperBoundRange = LVI.getConstantRange(Call->getArgOperand(2),
                                                         cast<CallInst>(Call));
      Value *UpperBound = nullptr;
      if (not UpperBoundRange.isFullSet()) {
        UpperBound = ConstantInt::get(DifferenceType,
                                      UpperBoundRange.getUpper());
      } else {
        UpperBound = Undef;
      }
      Call->setArgOperand(2, UpperBound);
    }
  }
  return true;
}

void ComputeStackAccessesBoundsPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LazyValueInfoWrapperPass>();
  AU.setPreservesCFG();
}

char ComputeStackAccessesBoundsPass::ID = 0;

using RegisterCSAB = RegisterPass<ComputeStackAccessesBoundsPass>;
static RegisterCSAB R("compute-stack-accesses-bounds",
                      "Compute Stack Accesses Bounds Pass");
