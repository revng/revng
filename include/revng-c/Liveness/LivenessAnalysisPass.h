#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/Pass.h"

#include "LivenessAnalysis.h"

struct LivenessAnalysisPass : public llvm::FunctionPass {

  static char ID;

  LivenessAnalysisPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  const LivenessAnalysis::LivenessMap &getLiveOut() const { return LiveOut; };

protected:
  LivenessAnalysis::LivenessMap LiveOut;
};
