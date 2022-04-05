/// \file CollectFunctionsFromCalleesPass.cpp
/// \brief Collect the function entry points from the callees.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromCalleesPass.h"

using namespace llvm;

char CollectFunctionsFromCalleesWrapperPass::ID = 0;

using Register = RegisterPass<CollectFunctionsFromCalleesWrapperPass>;
static Register Y("collect-functions-from-callees",
                  "Functions from callees collection pass",
                  true,
                  true);

static Logger<> Log("functions-from-callees-collection");

static void collectFunctionsFromCallees(Module &M,
                                        GeneratedCodeBasicInfo &GCBI,
                                        model::Binary &Binary) {
  Function &Root = *M.getFunction("root");

  // Static symbols have already been registered during lifting phase. Now
  // register all the other candidate entry points.
  for (BasicBlock &BB : Root) {
    if (getType(&BB) != BlockType::JumpTargetBlock)
      continue;

    MetaAddress Entry = GCBI.getJumpTarget(&BB);
    if (Binary.Functions.find(Entry) != Binary.Functions.end())
      continue;

    // Do not consider dynamic functions (e.g., PLT entries), they are imported
    // directly from EarlyFunctionAnalysis.
    if (not getDynamicSymbol(&BB).empty())
      continue;

    uint32_t Reasons = GCBI.getJTReasons(&BB);
    bool IsCallee = hasReason(Reasons, JTReason::Callee);

    if (IsCallee) {
      Binary.Functions[Entry].Type = model::FunctionType::Invalid;
      revng_log(Log, "Found function from callee: " << BB.getName().str());
    }
  }
}

bool CollectFunctionsFromCalleesWrapperPass::runOnModule(Module &M) {
  auto &LMWP = getAnalysis<LoadModelWrapperPass>().get();
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  collectFunctionsFromCallees(M, GCBI, *LMWP.getWriteableModel());
  return false;
}

PreservedAnalyses
CollectFunctionsFromCalleesPass::run(Module &M, ModuleAnalysisManager &MAM) {
  auto *LM = MAM.getCachedResult<LoadModelAnalysis>(M);
  if (!LM)
    return PreservedAnalyses::all();

  auto &GCBI = MAM.getResult<GeneratedCodeBasicInfoAnalysis>(M);

  collectFunctionsFromCallees(M, GCBI, *LM->getWriteableModel());
  return PreservedAnalyses::all();
}
