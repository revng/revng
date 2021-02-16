//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/IsolatedFunctions/IsolatedFunctions.h"

#include "DLAStep.h"
#include "DLATypeSystem.h"

using namespace dla;
using namespace llvm;

using StepT = CreateInterproceduralTypes;

bool StepT::runOnTypeSystem(LayoutTypeSystem &TS) {
  const Module &M = TS.getModule();
  const auto &Model = ModPass->getAnalysis<LoadModelPass>().getReadOnlyModel();
  for (const Function &F : M.functions()) {
    if (F.isIntrinsic() or not hasIsolatedFunction(Model, F))
      continue;
    revng_assert(not F.isVarArg());

    // Create the Function's return types
    auto FRetTypes = TS.getOrCreateLayoutTypes(F);

    // Create types for the Function's arguments
    for (const Argument &Arg : F.args()) {
      // Arguments can only be integers and pointers
      revng_assert(isa<IntegerType>(Arg.getType())
                   or isa<PointerType>(Arg.getType()));
      auto N = TS.getOrCreateLayoutTypes(Arg).size();
      // Given that arguments can only be integers or pointers, we should only
      // create a single LayoutType for each argument
      revng_assert(N == 1ULL);
    }

    for (const BasicBlock &B : F) {
      for (const Instruction &I : B) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {
          const Function *Callee = getCallee(Call);
          if (Callee->isIntrinsic() or not hasIsolatedFunction(Model, Callee)) {
            continue;
          }
          unsigned ArgNo = 0U;
          for (const Argument &FormalArg : Callee->args()) {
            Value *ActualArg = Call->getOperand(ArgNo);
            revng_assert(isa<IntegerType>(ActualArg->getType())
                         or isa<PointerType>(ActualArg->getType()));
            revng_assert(isa<IntegerType>(FormalArg.getType())
                         or isa<PointerType>(FormalArg.getType()));
            auto ActualTypes = TS.getOrCreateLayoutTypes(*ActualArg);
            auto FormalTypes = TS.getOrCreateLayoutTypes(FormalArg);
            revng_assert(1ULL == ActualTypes.size() == FormalTypes.size());
            auto FieldNum = FormalTypes.size();
            for (auto FieldId = 0ULL; FieldId < FieldNum; ++FieldId) {
              // Actual type inherits from formal type
              TS.addInheritanceLink(ActualTypes[FieldId].first,
                                    FormalTypes[FieldId].first);
            }
            ++ArgNo;
          }
        } else if (auto *PHI = dyn_cast<PHINode>(&I)) {
          revng_assert(isa<IntegerType>(PHI->getType())
                       or isa<PointerType>(PHI->getType()));
          auto PHITypes = TS.getOrCreateLayoutTypes(*PHI);
          for (const Use &Incoming : PHI->incoming_values()) {
            revng_assert(isa<IntegerType>(Incoming->getType())
                         or isa<PointerType>(Incoming->getType()));
            auto InTypes = TS.getOrCreateLayoutTypes(*Incoming.get());
            revng_assert(1ULL == PHITypes.size() == InTypes.size());
            auto FieldNum = PHITypes.size();
            for (auto FieldId = 0ULL; FieldId < FieldNum; ++FieldId) {
              // Incoming type inherits from PHI type
              TS.addInheritanceLink(InTypes[FieldId].first,
                                    PHITypes[FieldId].first);
            }
          }
        } else if (auto *RetI = dyn_cast<ReturnInst>(&I)) {
          if (Value *RetVal = RetI->getReturnValue()) {
            revng_assert(isa<StructType>(RetVal->getType())
                         or isa<IntegerType>(RetVal->getType())
                         or isa<PointerType>(RetVal->getType()));
            auto RetTypes = TS.getOrCreateLayoutTypes(*RetVal);
            revng_assert(RetTypes.size() == FRetTypes.size());
            auto FieldNum = RetTypes.size();
            for (auto FieldId = 0ULL; FieldId < FieldNum; ++FieldId) {
              // Return Operand type inherits from Function return type
              if (RetTypes[FieldId].first != nullptr)
                TS.addInheritanceLink(RetTypes[FieldId].first,
                                      FRetTypes[FieldId].first);
            }
          }
        }
      }
    }
  }
  if (VerifyLog.isEnabled())
    revng_assert(TS.verifyConsistency());
  return TS.getNumLayouts() != 0;
}
