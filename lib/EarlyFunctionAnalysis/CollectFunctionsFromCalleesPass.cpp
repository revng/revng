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
  std::vector<BasicBlock *> Functions;
  for (const auto &F : Binary.Functions)
    Functions.emplace_back(GCBI.getBlockAt(F.Entry));

  Function &Root = *M.getFunction("root");

  // Static symbols have already been registered during lifting phase. Now
  // register all the other candidate entry points.
  for (BasicBlock &BB : Root) {
    if (GCBI.getType(&BB) != BlockType::JumpTargetBlock)
      continue;

    auto It = llvm::find(Functions, &BB);
    if (It != Functions.end())
      continue;

    uint32_t Reasons = GCBI.getJTReasons(&BB);
    bool IsCallee = hasReason(Reasons, JTReason::Callee);

    if (IsCallee) {
      MetaAddress Entry = GCBI.getJumpTarget(&BB);
      Binary.Functions[Entry].Type = model::FunctionType::Invalid;
    }
  }

  for (const auto &F : Binary.Functions)
    revng_log(Log, "Found function from callees: " << F.CustomName.str());
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
