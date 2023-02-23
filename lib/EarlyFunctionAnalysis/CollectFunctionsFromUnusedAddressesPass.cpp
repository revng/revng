/// \file CollectFunctionsFromUnusedAddressesPass.cpp
/// \brief Collect the function entry points from unused addresses.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromUnusedAddressesPass.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"

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

  void run(FunctionMetadataCache &MDCache) {
    loadAllCFGs(MDCache);
    collectFunctionsFromUnusedAddresses();
  }

private:
  void loadAllCFGs(FunctionMetadataCache &MDCache) {
    for (auto &Function : Binary.Functions()) {
      llvm::BasicBlock *Entry = GCBI.getBlockAt(Function.Entry());
      llvm::Instruction *Term = Entry->getTerminator();
      auto *FMMDNode = Term->getMetadata(FunctionMetadataMDName);
      // CFG not serialized for this function? Skip it
      if (not FMMDNode)
        continue;

      const efa::FunctionMetadata &FM = MDCache.getFunctionMetadata(Entry);
      for (const efa::BasicBlock &Block : FM.ControlFlowGraph())
        VisitedBlocks.insert(Block.ID().start());
    }
  }

  void collectFunctionsFromUnusedAddresses() {
    using namespace llvm;
    Function &Root = *M.getFunction("root");

    for (BasicBlock &BB : Root) {
      if (getType(&BB) != BlockType::JumpTargetBlock)
        continue;

      MetaAddress Entry = getBasicBlockAddress(getJumpTargetBlock(&BB));
      if (Binary.Functions().find(Entry) != Binary.Functions().end())
        continue;

      uint32_t Reasons = GCBI.getJTReasons(&BB);
      bool IsUnusedGlobalData = hasReason(Reasons, JTReason::UnusedGlobalData);
      bool IsMemoryStore = hasReason(Reasons, JTReason::MemoryStore);
      bool IsPCStore = hasReason(Reasons, JTReason::PCStore);
      bool IsReturnAddress = hasReason(Reasons, JTReason::ReturnAddress);
      bool IsLoadAddress = hasReason(Reasons, JTReason::LoadAddress);
      bool IsNotPartOfOtherCFG = VisitedBlocks.count(Entry) == 0;

      // Do not consider addresses found in .rodata that are part of jump
      // tables of a function.
      if (not IsLoadAddress
          and (IsUnusedGlobalData
               or (IsMemoryStore and not IsPCStore and not IsReturnAddress))
          and IsNotPartOfOtherCFG) {
        // TODO: keep IsReturnAddress?
        // Consider addresses found in global data that have not been used or
        // addresses that are not return addresses and do not end up in the PC
        // directly.
        Binary.Functions()[Entry];
        revng_log(Log,
                  "Found function from unused addresses: "
                    << BB.getName().str());
      }
    }
  }

private:
  llvm::Module &M;
  GeneratedCodeBasicInfo &GCBI;
  model::Binary &Binary;
  SortedVector<MetaAddress> VisitedBlocks;
};

bool CFFUAWrapperPass::runOnModule(llvm::Module &M) {
  auto &LMWP = getAnalysis<LoadModelWrapperPass>().get();
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  CFFUAImpl Impl(M, GCBI, *LMWP.getWriteableModel());
  Impl.run(getAnalysis<FunctionMetadataCachePass>().get());
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
  Impl.run(*MAM.getCachedResult<FunctionMetadataCacheAnalysis>(M));
  return llvm::PreservedAnalyses::all();
}

void CFFUAWrapperPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<FunctionMetadataCachePass>();
  AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
}
