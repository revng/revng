//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// local includes
#include "revng-c/Liveness/LivenessAnalysisPass.h"

using namespace llvm;

bool LivenessAnalysisPass::runOnFunction(Function &F) {
    LivenessAnalysis::Analysis LA(F);
    LA.initialize();
    LA.run();
    LiveOut = LA.extractLiveOut();
    return false;
  }

char LivenessAnalysisPass::ID = 0;

static RegisterPass<LivenessAnalysisPass> X("liveness", "Liveness Analysis");
