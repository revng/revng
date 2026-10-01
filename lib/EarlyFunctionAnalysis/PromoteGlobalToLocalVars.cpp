/// Promote CSVs in form of global variables to local variables.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"

#include "revng/EarlyFunctionAnalysis/PromoteGlobalToLocalVars.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OpaqueRegisterUser.h"

using namespace llvm;

llvm::PreservedAnalyses
PromoteGlobalToLocalPass::run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {

  // Collect the CSVs used by the current function.
  std::map<GlobalVariable *, Value *> CSVMap;
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
        if (not ShouldPromote or ShouldPromote(*CSV))
          CSVMap.try_emplace(CSV);
    }
  }

  revng::IRBuilder Builder(&F.getEntryBlock().front());

  llvm::SmallVector<GlobalVariable *> CSVsToPromote;
  for (GlobalVariable *CSV : llvm::make_first_range(CSVMap))
    CSVsToPromote.push_back(CSV);
  CSVInFunctionReplacer Replacer(CSVsToPromote, F);

  // Create an equivalent local variable, replace all the uses of the CSV.
  for (GlobalVariable *CSV : Replacer.csvs()) {
    auto *CSVTy = CSV->getValueType();
    auto *Alloca = Builder.CreateAlloca(CSVTy, nullptr, CSV->getName());
    Replacer.replaceCSVWithAlloca(CSV, Alloca);

    CSVMap[CSV] = Alloca;
  }

  // Load all the CSVs and store their value onto the local variables.
  for (const auto &[CSV, Alloca] : CSVMap)
    Builder.CreateStore(Builder.createLoad(CSV), Alloca);

  return PreservedAnalyses::none();
}
