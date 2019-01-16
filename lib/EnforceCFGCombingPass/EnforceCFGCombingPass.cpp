//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/Pass.h>

// local librariesincludes
#include "revng-c/RestructureCFGPass/RegionCFGTree.h"

// local includes
#include "BasicBlockViewAnalysis.h"
#include "EnforceCFGCombingPass.h"

using namespace llvm;

bool EnforceCFGCombingPass::runOnFunction(Function &F) {
  auto &LA = getAnalysis<LivenessAnalysisPass>();
  const LivenessAnalysis::LivenessMap &LiveIn = LA.getLiveIn();

  auto &RestructurePass = getAnalysis<RestructureCFG>();
  CFG &RCFGT = RestructurePass.getRCT();

  // TODO: Perform preprocessing on RCFGT to ensure that each node with more
  // than one successor only has dummy successors. If that's not true, inject
  // dummy successors when necessary.

  BasicBlockViewAnalysis::Analysis BBViewAnalysis(RCFGT, F);
  BBViewAnalysis.initialize();
  BBViewAnalysis.run();
  return true;
}

char EnforceCFGCombingPass::ID = 0;

static RegisterPass<EnforceCFGCombingPass>
X("enforce-combing",
  "Enforce Combing on the Control Flow Graph of all Functions", false, false);
