//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include <llvm/Pass.h>
#include <llvm/Support/raw_ostream.h>

// local includes
#include "LivenessAnalysis.h"

using namespace llvm;

struct LivenessAnalysisPass : public FunctionPass {
  static char ID;

  LivenessAnalysisPass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    LivenessAnalysis::Analysis LA(F);
    LA.initialize();
    LA.run();
    return false;
  }

};

char LivenessAnalysisPass::ID = 0;

static RegisterPass<LivenessAnalysisPass> X("liveness", "Liveness Analysis", false, false);
