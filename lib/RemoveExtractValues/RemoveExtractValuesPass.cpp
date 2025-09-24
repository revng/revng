//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/Local.h"

#include "revng/Model/FunctionTags.h"
#include "revng/RemoveExtractValues/RemoveExtractValuesPass.h"
#include "revng/Support/OpaqueFunctionsPool.h"

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

  // Create a pool of functions with the same behavior: we will need a different
  // function for each different struct
  auto OpaqueEVPool = FunctionTags::OpaqueExtractValue.getPool(*F.getParent());

  llvm::LLVMContext &LLVMCtx = F.getContext();

  // TODO: checks are only omitted here because of unit tests.
  revng::NonDebugInfoCheckingIRBuilder Builder(LLVMCtx);

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
    FunctionTags::TypePair Key = { I->getType(),
                                   I->getAggregateOperand()->getType() };
    auto *ExtractValueFunction = OpaqueEVPool.get(Key,
                                                  EVFunctionType,
                                                  "OpaqueExtractvalue");

    // Emit a call to the new function
    CallInst *InjectedCall = Builder.CreateCall(ExtractValueFunction,
                                                ArgValues);
    I->replaceAllUsesWith(InjectedCall);
    InjectedCall->copyMetadata(*I);
    llvm::RecursivelyDeleteTriviallyDeadInstructions(I);
  }

  return true;
}
