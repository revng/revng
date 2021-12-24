/// \file NonForcedCFEPCollectionPass.cpp
/// \brief Collect the non-forced Candidate Function Entry Points.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromUnusedAddressesPass.h"

using namespace llvm;

char CollectFunctionsFromUnusedAddressesWrapperPass::ID = 0;

using Register = RegisterPass<CollectFunctionsFromUnusedAddressesWrapperPass>;
static Register Y("collect-functions-from-unused-addresses",
                  "Functions from unused addresses collection pass",
                  true,
                  true);

static Logger<> Log("functions-from-unused-addresses-collection");

static void collectFunctionsFromUnusedAddresses(Module &M,
                                                GeneratedCodeBasicInfo &GCBI,
                                                model::Binary &Binary) {
  std::vector<BasicBlock *> Functions;
  Function &Root = *M.getFunction("root");

  for (BasicBlock &BB : Root) {
    if (GCBI.getType(&BB) != BlockType::JumpTargetBlock)
      continue;

    uint32_t Reasons = GCBI.getJTReasons(&BB);
    bool IsUnusedGlobalData = hasReason(Reasons, JTReason::UnusedGlobalData);
    bool IsMemoryStore = hasReason(Reasons, JTReason::MemoryStore);
    bool IsPCStore = hasReason(Reasons, JTReason::PCStore);
    bool IsReturnAddress = hasReason(Reasons, JTReason::ReturnAddress);
    bool IsLoadAddress = hasReason(Reasons, JTReason::LoadAddress);

    if (not IsLoadAddress
        and (IsUnusedGlobalData
             || (IsMemoryStore and not IsPCStore and not IsReturnAddress))) {
      // TODO: keep IsReturnAddress?
      // Consider addresses found in global data that have not been used or
      // addresses that are not return addresses and do not end up in the PC
      // directly.
      MetaAddress Entry = GCBI.getJumpTarget(&BB);
      Binary.Functions[Entry].Type = model::FunctionType::Invalid;
      Functions.emplace_back(&BB);
    }
  }

  for (const auto &BB : Functions)
    revng_log(Log,
              "Found function from unused addresses: " << BB->getName().str());
}

bool CollectFunctionsFromUnusedAddressesWrapperPass::runOnModule(Module &M) {
  auto &LMWP = getAnalysis<LoadModelWrapperPass>().get();
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  collectFunctionsFromUnusedAddresses(M, GCBI, *LMWP.getWriteableModel());
  return false;
}

PreservedAnalyses
CollectFunctionsFromUnusedAddressesPass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  auto *LM = MAM.getCachedResult<LoadModelAnalysis>(M);
  if (!LM)
    return PreservedAnalyses::all();

  auto &GCBI = MAM.getResult<GeneratedCodeBasicInfoAnalysis>(M);

  collectFunctionsFromUnusedAddresses(M, GCBI, *LM->getWriteableModel());
  return PreservedAnalyses::all();
}
