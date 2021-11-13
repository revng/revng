//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"

#include "revng/Model/Binary.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/DataLayoutAnalysis/DLATypeSystem.h"

#include "../DLAModelFuncHelpers.h"
#include "DLATypeSystemBuilder.h"

using namespace dla;
using namespace llvm;

using TSBuilder = DLATypeSystemLLVMBuilder;

///\brief Given an llvm Function, return its prototype in the model.
static const model::Type *
getPrototype(const llvm::Function &F, const model::Binary &Model) {
  auto MetaAddr = getMetaAddress(&F);
  auto &ModelFunc = Model.Functions.at(MetaAddr);
  return ModelFunc.Prototype.get();
}

bool TSBuilder::createInterproceduralTypes(llvm::Module &M,
                                           const model::Binary &Model) {
  for (const Function &F : M.functions()) {
    auto FTags = FunctionTags::TagsSet::from(&F);
    if (F.isIntrinsic() or not FTags.contains(FunctionTags::Lifted))
      continue;
    revng_assert(not F.isVarArg());

    // Check if a function with the same prototype has already been visited
    auto *Prototype = getPrototype(F, Model);
    FuncOrCallInst FuncWithSameProto;
    auto It = VisitedPrototypes.find(Prototype);
    if (It == VisitedPrototypes.end())
      VisitedPrototypes.insert({ Prototype, &F });
    else
      FuncWithSameProto = It->second;

    // Create the Function's return types
    auto FRetTypes = getOrCreateLayoutTypes(F);
    // Add equality links between return values of function with the same
    // prototype
    if (not FuncWithSameProto.isNull()) {
      auto OtherRetVals = getLayoutTypes(*FuncWithSameProto.getVal());
      revng_assert(FRetTypes.size() == OtherRetVals.size());
      for (auto [N1, N2] : llvm::zip(OtherRetVals, FRetTypes))
        TS.addEqualityLink(N1, N2.first);
    }

    revng_assert(FuncWithSameProto.isNull()
                 or F.arg_size() == FuncWithSameProto.arg_size());

    // Create types for the Function's arguments
    for (const auto &Arg : llvm::enumerate(F.args())) {
      // Arguments can only be integers and pointers
      auto &ArgVal = Arg.value();
      revng_assert(isa<IntegerType>(ArgVal.getType())
                   or isa<PointerType>(ArgVal.getType()));
      auto [ArgNode, _] = getOrCreateLayoutType(&ArgVal);
      revng_assert(ArgNode);

      // If there is already a Function with the same prototype, add equality
      // edges between args
      if (not FuncWithSameProto.isNull()) {
        auto &OtherArg = *(FuncWithSameProto.getArg(Arg.index()));
        auto *OtherArgNode = getLayoutType(&OtherArg);
        revng_assert(OtherArgNode);
        TS.addEqualityLink(ArgNode, OtherArgNode);
      }
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
