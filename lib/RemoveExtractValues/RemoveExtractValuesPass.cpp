//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/LoadModelPass.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/RemoveExtractValues/RemoveExtractValuesPass.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

using namespace llvm;

char RemoveExtractValues::ID = 0;
using Reg = RegisterPass<RemoveExtractValues>;
static Reg X("remove-extractvalues",
             "Substitute extractvalues with opaque calls so that they don't "
             "get optimized",
             true,
             true);

void RemoveExtractValues::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<FunctionMetadataCachePass>();
}

bool RemoveExtractValues::runOnFunction(llvm::Function &F) {
  using namespace llvm;

  // Collect all ExtractValues
  SmallVector<ExtractValueInst *, 16> ToReplace;
  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *ExtractVal = llvm::dyn_cast<llvm::ExtractValueInst>(&I))
        ToReplace.push_back(ExtractVal);

  if (ToReplace.empty())
    return false;

  auto &Cache = getAnalysis<FunctionMetadataCachePass>().get();
  auto &Model = getAnalysis<LoadModelWrapperPass>().get().getReadOnlyModel();

  // Create a pool of functions with the same behavior: we will need a different
  // function for each different struct
  OpaqueFunctionsPool<TypePair> OpaqueEVPool(F.getParent(),
                                             /* PurgeOnDestruction */ false);
  initOpaqueEVPool(OpaqueEVPool, F.getParent());

  llvm::LLVMContext &LLVMCtx = F.getContext();
  IRBuilder<> Builder(LLVMCtx);
  for (ExtractValueInst *I : ToReplace) {
    Builder.SetInsertPoint(I);

    // Collect arguments of the ExtractValue
    SmallVector<Value *, 8> ArgValues = { I->getAggregateOperand() };
    revng_assert(I->getNumIndices() == 1);
    for (auto Idx : I->indices()) {
      auto *IndexVal = ConstantInt::get(IntegerType::getInt64Ty(LLVMCtx), Idx);
      ArgValues.push_back(IndexVal);
    }

    // Get or generate the function
    auto *EVFunctionType = getOpaqueEVFunctionType(I);
    const TypePair &Key = { I->getType(), I->getAggregateOperand()->getType() };
    auto *ExtractValueFunction = OpaqueEVPool.get(Key,
                                                  EVFunctionType,
                                                  "OpaqueExtractvalue");

    // Emit a call to the new function
    CallInst *InjectedCall = Builder.CreateCall(ExtractValueFunction,
                                                ArgValues);
    using PTMLCBuilder = ptml::PTMLCBuilder;
    PTMLCBuilder B(/*GeneratePlainC*/ true);
    std::string StructName;
    revng_assert(isa<CallInst>(I->getAggregateOperand()));
    auto *CallReturningAggregate = cast<CallInst>(I->getAggregateOperand());
    if (isCallToIsolatedFunction(CallReturningAggregate)) {
      auto Prototype = Cache.getCallSitePrototype(*Model,
                                                  CallReturningAggregate);
      revng_assert(Prototype.get());
      StructName = std::string(getReturnTypeName(*Prototype.get(), B));
    } else {
      // If it is not a isolated function, it must be a helper function.
      auto CalledFunction = CallReturningAggregate->getCalledFunction();
      revng_assert(CalledFunction);
      StructName = getReturnTypeLocationDefinition(CalledFunction, B);
    }

    auto AggregateType = I->getAggregateOperand()->getType();
    auto *TheStructType = llvm::cast<llvm::StructType>(AggregateType);
    if (not TheStructType->isLiteral())
      TheStructType->setName(StructName);

    I->replaceAllUsesWith(InjectedCall);
    InjectedCall->copyMetadata(*I);
    llvm::RecursivelyDeleteTriviallyDeadInstructions(I);
  }

  return true;
}
