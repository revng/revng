/// \file PromoteGlobalToLocalVars.cpp
/// \brief Promote CSVs in form of global variables to local variables.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"

#include "revng/BasicAnalyses/PromoteGlobalToLocalVars.h"
#include "revng/Support/IRHelpers.h"

llvm::PreservedAnalyses
PromoteGlobalToLocalPass::run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
  using namespace llvm;

  // Collect the CSVs used by the current function.
  std::set<GlobalVariable *> CSVs;
  for (auto &BB : F) {
    for (auto &I : BB) {
      Value *Pointer = nullptr;
      if (auto *Load = dyn_cast<LoadInst>(&I))
        Pointer = skipCasts(Load->getPointerOperand());
      else if (auto *Store = dyn_cast<StoreInst>(&I))
        Pointer = skipCasts(Store->getPointerOperand());
      else
        continue;

      if (auto *CSV = dyn_cast_or_null<GlobalVariable>(Pointer))
        CSVs.insert(CSV);
    }
  }

  // Create an equivalent local variable, replace all the uses of the CSV,
  // and assign it to its local counterpart.
  IRBuilder<> Builder(&F.getEntryBlock().front());
  for (auto *CSV : CSVs) {
    auto *CSVTy = CSV->getType()->getPointerElementType();
    auto *Alloca = Builder.CreateAlloca(CSVTy, nullptr, CSV->getName());

    replaceAllUsesInFunctionWith(&F, CSV, Alloca);

    Builder.CreateStore(Builder.CreateLoad(CSV), Alloca);
  }

  return PreservedAnalyses::none();
}
