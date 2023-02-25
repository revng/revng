//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueFunctionsPool.h"

#include "revng-c/RemoveExtractValues/RestoreExtractValuesPass.h"
#include "revng-c/Support/FunctionTags.h"

using namespace llvm;

char RestoreExtractValues::ID = 0;
using Reg = RegisterPass<RestoreExtractValues>;
static Reg X("restore-extractvalues",
             "Substitute extractvalue() calls with actual extractvalues",
             true,
             true);

static bool isOpaqueExtractValue(llvm::Instruction *I) {
  if (auto *Call = llvm::dyn_cast<llvm::CallInst>(I))
    if (auto *CalledFunc = Call->getCalledFunction())
      return FunctionTags::OpaqueExtractValue.isTagOf(CalledFunc);

  return false;
}

bool RestoreExtractValues::runOnFunction(llvm::Function &F) {
  using namespace llvm;

  // Collect all ExtractValues
  SmallVector<CallInst *, 16> ToReplace;
  for (auto &BB : F)
    for (auto &I : BB)
      if (isOpaqueExtractValue(&I))
        ToReplace.push_back(llvm::cast<CallInst>(&I));

  if (ToReplace.empty())
    return false;

  llvm::LLVMContext &LLVMCtx = F.getContext();
  IRBuilder<> Builder(LLVMCtx);
  for (CallInst *I : ToReplace) {
    Builder.SetInsertPoint(I);

    // Collect the indices values
    SmallVector<unsigned int, 8> Indexes;
    for (auto &Op : llvm::drop_begin(I->args())) {
      uint64_t Idx = llvm::cast<ConstantInt>(&Op)->getValue().getLimitedValue();
      Indexes.push_back(Idx);
    }

    // Emit extractvalue
    auto *Injected = Builder.CreateExtractValue(I->getArgOperand(0), Indexes);

    I->replaceAllUsesWith(Injected);
    I->eraseFromParent();
  }

  return true;
}
