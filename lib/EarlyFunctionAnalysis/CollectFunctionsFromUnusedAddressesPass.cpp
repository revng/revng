/// \file CollectFunctionsFromUnusedAddressesPass.cpp
/// \brief Collect the function entry points from unused addresses.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromUnusedAddressesPass.h"

using CFFUAWrapperPass = CollectFunctionsFromUnusedAddressesWrapperPass;
char CFFUAWrapperPass::ID = 0;

using Register = llvm::RegisterPass<CFFUAWrapperPass>;
static Register Y("collect-functions-from-unused-addresses",
                  "Functions from unused addresses collection pass",
                  true,
                  true);

static Logger<> Log("functions-from-unused-addresses-collection");

class CFFUAImpl {
public:
  CFFUAImpl(llvm::Module &M,
            GeneratedCodeBasicInfo &GCBI,
            model::Binary &Binary) :
    M(M), GCBI(GCBI), Binary(Binary) {}

  void run();

private:
  void collectFunctionsFromUnusedAddresses();

private:
  llvm::Module &M;
  GeneratedCodeBasicInfo &GCBI;
  model::Binary &Binary;
};

void CFFUAImpl::collectFunctionsFromUnusedAddresses() {
  using namespace llvm;
  Function &Root = *M.getFunction("root");

  for (BasicBlock &BB : Root) {
    if (getType(&BB) != BlockType::JumpTargetBlock)
      continue;

    MetaAddress Entry = GCBI.getJumpTarget(&BB);
    if (Binary.Functions.find(Entry) != Binary.Functions.end())
      continue;

    uint32_t Reasons = GCBI.getJTReasons(&BB);
    bool IsUnusedGlobalData = hasReason(Reasons, JTReason::UnusedGlobalData);
    bool IsMemoryStore = hasReason(Reasons, JTReason::MemoryStore);
    bool IsPCStore = hasReason(Reasons, JTReason::PCStore);
    bool IsReturnAddress = hasReason(Reasons, JTReason::ReturnAddress);
    bool IsLoadAddress = hasReason(Reasons, JTReason::LoadAddress);

    // Do not consider addresses found in .rodata that are part of jump
    // tables of a function.
    if (not IsLoadAddress
        and (IsUnusedGlobalData
             or (IsMemoryStore and not IsPCStore and not IsReturnAddress))) {
      // TODO: keep IsReturnAddress?
      // Consider addresses found in global data that have not been used or
      // addresses that are not return addresses and do not end up in the PC
      // directly.
      Binary.Functions[Entry].Type = model::FunctionType::Invalid;
      revng_log(Log,
                "Found function from unused addresses: " << BB.getName().str());
    }
  }
}

void CFFUAImpl::run() {
  collectFunctionsFromUnusedAddresses();
}

bool CFFUAWrapperPass::runOnModule(llvm::Module &M) {
  auto &LMWP = getAnalysis<LoadModelWrapperPass>().get();
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  CFFUAImpl Impl(M, GCBI, *LMWP.getWriteableModel());
  Impl.run();
  return false;
}

llvm::PreservedAnalyses
CollectFunctionsFromUnusedAddressesPass::run(llvm::Module &M,
                                             llvm::ModuleAnalysisManager &MAM) {
  auto *LM = MAM.getCachedResult<LoadModelAnalysis>(M);
  if (!LM)
    return llvm::PreservedAnalyses::all();

  auto &GCBI = MAM.getResult<GeneratedCodeBasicInfoAnalysis>(M);

  CFFUAImpl Impl(M, GCBI, *LM->getWriteableModel());
  Impl.run();
  return llvm::PreservedAnalyses::all();
}
