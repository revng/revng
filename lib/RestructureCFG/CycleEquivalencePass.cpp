//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng-c/RestructureCFG/CycleEquivalencePass.h"
#include "revng-c/RestructureCFG/CycleEquivalenceResult.h"

using namespace llvm;

char CycleEquivalencePass::ID = 0;

static constexpr const char *Flag = "cycle-equivalence";
using Reg = llvm::RegisterPass<CycleEquivalencePass>;
static Reg X(Flag, "Cycle Equivalence Pass");

bool CycleEquivalencePass::runOnFunction(llvm::Function &F) {

  EdgeToCycleEquivalenceClassIDMap = getEdgeToCycleEquivalenceClassIDMap(&F);

  // The goal of `CycleEquivalencePass` is to just perform an analysis
  // computation, this operation should not perform a change of the IR
  return false;
}

void CycleEquivalencePass::getAnalysisUsage(llvm::AnalysisUsage &AU) const {

  // This is a read only analysis, that does not touch the IR
  AU.setPreservesAll();
}

const CycleEquivalenceAnalysis<Function *>::CycleEquivalenceResult &
CycleEquivalencePass::getResult() {
  return EdgeToCycleEquivalenceClassIDMap;
}
