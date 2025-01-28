/// \file CollectFunctionsFromUnusedAddressesPass.cpp
/// Collect the function entry points from unused addresses.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "boost/icl/interval_set.hpp"
#include "boost/icl/right_open_interval.hpp"

#include "llvm/IR/Module.h"

#include "revng/EarlyFunctionAnalysis/CollectFunctionsFromUnusedAddressesPass.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Lift/Lift.h"

using namespace llvm;

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

  void run(ControlFlowGraphCache &MDCache) {
    loadAllCFGs(MDCache);
    collectFunctionsFromUnusedAddresses();
  }

private:
  void loadAllCFGs(ControlFlowGraphCache &MDCache) {
    for (auto &Function : Binary.Functions()) {
      llvm::BasicBlock *Entry = GCBI.getBlockAt(Function.Entry());
      llvm::Instruction *Term = Entry->getTerminator();

      const efa::ControlFlowGraph &FM = MDCache.getControlFlowGraph(Function
                                                                      .Entry());
      for (const efa::BasicBlock &Block : FM.Blocks()) {
        auto Start = Block.ID().start();
        auto End = Block.End();
        revng_log(Log,
                  "Registering as used range [" << Start.toString() << ", "
                                                << End.toString() << ")");
        UsedRanges += interval::right_open(Start, End);
      }
    }
  }

  void collectFunctionsFromUnusedAddresses() {
    using namespace llvm;
    Function &Root = *M.getFunction("root");

    for (BasicBlock &BB : Root) {
      if (getType(&BB) != BlockType::JumpTargetBlock)
        continue;

      MetaAddress Entry = getBasicBlockAddress(getJumpTargetBlock(&BB));
      if (Binary.Functions().tryGet(Entry) != nullptr)
        continue;

      uint32_t Reasons = GCBI.getJTReasons(&BB);
      bool IsSimpleLiteral = hasReason(Reasons, JTReason::SimpleLiteral);
      bool IsUnusedGlobalData = hasReason(Reasons, JTReason::UnusedGlobalData);
      bool IsDirectJump = hasReason(Reasons, JTReason::DirectJump);
      bool IsMemoryStore = hasReason(Reasons, JTReason::MemoryStore);
      bool IsPCStore = hasReason(Reasons, JTReason::PCStore);
      bool IsReturnAddress = hasReason(Reasons, JTReason::ReturnAddress);
      bool IsLoadAddress = hasReason(Reasons, JTReason::LoadAddress);
      bool IsPartOfOtherCFG = UsedRanges.find(Entry) != UsedRanges.end();

      revng_log(Log,
                Entry.toString()
                  << "\n"
                  << "  IsSimpleLiteral: " << IsSimpleLiteral << "\n"
                  << "  IsUnusedGlobalData: " << IsUnusedGlobalData << "\n"
                  << "  IsDirectJump: " << IsDirectJump << "\n"
                  << "  IsMemoryStore: " << IsMemoryStore << "\n"
                  << "  IsPCStore: " << IsPCStore << "\n"
                  << "  IsReturnAddress: " << IsReturnAddress << "\n"
                  << "  IsLoadAddress: " << IsLoadAddress << "\n"
                  << "  IsPartOfOtherCFG: " << IsPartOfOtherCFG << "\n");

      // Look for simple literals, unused global data or address stored in
      // memory that are never written directly to the PC. Also, ignore code for
      // which we already have a direct jump or is part of the CFG of another
      // function. Also, ignore addresses that are return addresses.
      bool Interesting = (IsSimpleLiteral or IsUnusedGlobalData
                          or (IsMemoryStore and not IsPCStore));
      bool Ignore = IsPartOfOtherCFG or IsDirectJump or IsReturnAddress;
      if (Interesting and not Ignore) {

        if (IsLoadAddress) {
          revng_log(Log,
                    "Ignoring " << Entry.toString()
                                << " since it's target of a memory operation");

        } else {
          if (IsMemoryStore and not IsUnusedGlobalData) {
            revng_log(Log,
                      "Creating function due to memory store at "
                        << Entry.toString());
          }

          Binary.Functions()[Entry];
          revng_log(Log,
                    "Found function from unused addresses: "
                      << BB.getName().str());
        }
      }
    }
  }

private:
  using interval_set = boost::icl::interval_set<MetaAddress>;
  using interval = boost::icl::interval<MetaAddress>;

private:
  llvm::Module &M;
  GeneratedCodeBasicInfo &GCBI;
  model::Binary &Binary;
  interval_set UsedRanges;
};

bool CFFUAWrapperPass::runOnModule(llvm::Module &M) {
  auto &LMWP = getAnalysis<LoadModelWrapperPass>().get();
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();

  CFFUAImpl Impl(M, GCBI, *LMWP.getWriteableModel());
  Impl.run(getAnalysis<ControlFlowGraphCachePass>().get());
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
  Impl.run(*MAM.getCachedResult<ControlFlowGraphCacheAnalysis>(M));
  return llvm::PreservedAnalyses::all();
}

void CFFUAWrapperPass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<LoadModelWrapperPass>();
  AU.addRequired<ControlFlowGraphCachePass>();
  AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
}
