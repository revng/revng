//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "DLATypeSystemBuilder.h"

using namespace dla;
using namespace llvm;

bool DLATypeSystemLLVMBuilder::createInterproceduralTypes(llvm::Module &M) {
  for (const Function &F : M.functions()) {
    auto FTags = FunctionTags::TagsSet::from(&F);
    if (F.isIntrinsic() or not FTags.contains(FunctionTags::Lifted))
      continue;
    revng_assert(not F.isVarArg());

    // Create the Function's return types
    auto FRetTypes = getOrCreateLayoutTypes(F);

    // Create types for the Function's arguments
    for (const Argument &Arg : F.args()) {
      // Arguments can only be integers and pointers
      revng_assert(isa<IntegerType>(Arg.getType())
                   or isa<PointerType>(Arg.getType()));
      auto N = getOrCreateLayoutTypes(Arg).size();
      // Given that arguments can only be integers or pointers, we should only
      // create a single LayoutType for each argument
      revng_assert(N == 1ULL);
    }

    for (const BasicBlock &B : F) {
      for (const Instruction &I : B) {
        if (auto *Call = dyn_cast<CallInst>(&I)) {

          const Function *Callee = getCallee(Call);

          // TODO: this case will need to be handled properly to be able to
          // infer types from calls to dynamic functions.
          // Calls to dynamic functions at the moment don't have a callee,
          // because the callees are generated with a bunch of pointer
          // arithmetic from integer constants.
          if (not Callee)
            continue;

          auto CTags = FunctionTags::TagsSet::from(Callee);
          if (Callee->isIntrinsic() or not CTags.contains(FunctionTags::Lifted))
            continue;

          unsigned ArgNo = 0U;
          for (const Argument &FormalArg : Callee->args()) {
            Value *ActualArg = Call->getOperand(ArgNo);
            revng_assert(isa<IntegerType>(ActualArg->getType())
                         or isa<PointerType>(ActualArg->getType()));
            revng_assert(isa<IntegerType>(FormalArg.getType())
                         or isa<PointerType>(FormalArg.getType()));
            auto ActualTypes = getOrCreateLayoutTypes(*ActualArg);
            auto FormalTypes = getOrCreateLayoutTypes(FormalArg);
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
                       or isa<PointerType>(PHI->getType())
                       or isa<StructType>(PHI->getType()));
          auto PHITypes = getOrCreateLayoutTypes(*PHI);
          for (const Use &Incoming : PHI->incoming_values()) {
            revng_assert(isa<IntegerType>(Incoming->getType())
                         or isa<PointerType>(Incoming->getType())
                         or isa<StructType>(Incoming->getType()));
            auto InTypes = getOrCreateLayoutTypes(*Incoming.get());
            revng_assert(PHITypes.size() == InTypes.size());
            revng_assert((PHITypes.size() == 1ULL)
                         or isa<StructType>(PHI->getType()));
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
            auto RetTypes = getOrCreateLayoutTypes(*RetVal);
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
