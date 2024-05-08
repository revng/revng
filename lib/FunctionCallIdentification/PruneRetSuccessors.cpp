/// \file PruneRetSuccessors.cpp
/// Remove successors from return instructions

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/FunctionCallIdentification/PruneRetSuccessors.h"
#include "revng/Support/Debug.h"

using namespace llvm;

char PruneRetSuccessors::ID = 0;
using Register = RegisterPass<PruneRetSuccessors>;
static Register X("prs", "Prune Ret Successors", true, true);

struct SuccessorsList {
  bool AnyPC = false;
  bool UnexpectedPC = false;
  bool Other = false;
  std::set<MetaAddress> Addresses;

  bool operator==(const SuccessorsList &Other) const = default;

  static SuccessorsList other() {
    SuccessorsList Result;
    Result.Other = true;
    return Result;
  }

  bool hasSuccessors() const {
    return AnyPC or UnexpectedPC or Other or Addresses.size() != 0;
  }

  void dump() const debug_function { dump(dbg); }

  template<typename O>
  void dump(O &Output) const {
    Output << "AnyPC: " << AnyPC << "\n";
    Output << "UnexpectedPC: " << UnexpectedPC << "\n";
    Output << "Other: " << Other << "\n";
    Output << "Addresses:\n";
    for (const MetaAddress &Address : Addresses) {
      Output << "  ";
      Address.dump(Output);
      Output << "\n";
    }
  }
};

static SuccessorsList getSuccessors(GeneratedCodeBasicInfo &GCBI,
                                    BasicBlock *BB) {
  bool IsRoot = BB->getParent() == GCBI.root();

  SuccessorsList Result;

  df_iterator_default_set<BasicBlock *> Visited;

  if (IsRoot) {
    Visited.insert(GCBI.anyPC());
    Visited.insert(GCBI.unexpectedPC());
  }

  for (BasicBlock *Block : depth_first_ext(BB, Visited)) {
    for (BasicBlock *Successor : successors(Block)) {
      revng_assert(Successor != GCBI.dispatcher());

      MetaAddress Address = getBasicBlockID(Successor).start();
      const auto IBDHB = BlockType::IndirectBranchDispatcherHelperBlock;
      if (Address.isValid()) {
        Visited.insert(Successor);
        Result.Addresses.insert(Address);
      } else if (IsRoot and Successor == GCBI.anyPC()) {
        Result.AnyPC = true;
      } else if (IsRoot and Successor == GCBI.unexpectedPC()) {
        Result.UnexpectedPC = true;
      } else if (getType(Successor) == IBDHB) {
        // Ignore
      } else {
        Result.Other = true;
      }
    }
  }

  return Result;
}

bool PruneRetSuccessors::runOnModule(llvm::Module &M) {
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto &FCI = getAnalysis<FunctionCallIdentification>();

  for (BasicBlock &BB : *GCBI.root()) {
    if (not GCBI.isTranslated(&BB)
        or BB.getTerminator()->getNumSuccessors() < 2)
      continue;

    auto Successors = getSuccessors(GCBI, &BB);
    if (not Successors.UnexpectedPC or Successors.Other)
      continue;

    revng_assert(not Successors.AnyPC);
    bool AllFallthrough = true;
    for (MetaAddress SuccessorMA : Successors.Addresses) {
      if (not FCI.isFallthrough(SuccessorMA)) {
        AllFallthrough = false;
      }
    }

    if (AllFallthrough) {
      Instruction *OldTerminator = BB.getTerminator();
      auto *NewTerminator = BranchInst::Create(GCBI.anyPC(), &BB);
      NewTerminator->copyMetadata(*OldTerminator);
      eraseFromParent(OldTerminator);
    }
  }

  return true;
}
