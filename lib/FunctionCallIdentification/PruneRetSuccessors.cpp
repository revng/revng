/// \file PruneRetSuccessors.cpp
/// \brief Remove successors from return instructions

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/FunctionCallIdentification/PruneRetSuccessors.h"
#include "revng/Support/Debug.h"

using namespace llvm;

char PruneRetSuccessors::ID = 0;
using Register = RegisterPass<PruneRetSuccessors>;
static Register X("prs", "Prune Ret Successors", true, true);

bool PruneRetSuccessors::runOnModule(llvm::Module &M) {
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto &FCI = getAnalysis<FunctionCallIdentification>();

  for (BasicBlock &BB : *GCBI.root()) {
    if (not GCBI.isTranslated(&BB)
        or BB.getTerminator()->getNumSuccessors() < 2)
      continue;

    auto Successors = GCBI.getSuccessors(&BB);
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
