//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef REVNGC_LIVENESS_ANALYSIS_PASS_H
#define REVNGC_LIVENESS_ANALYSIS_PASS_H

// LLVM includes
#include <llvm/Pass.h>

// local includes
#include "LivenessAnalysis.h"

struct LivenessAnalysisPass : public llvm::FunctionPass {

  static char ID;

  LivenessAnalysisPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  const LivenessAnalysis::LivenessMap &getLiveOut() const { return LiveOut; };

protected:
  LivenessAnalysis::LivenessMap LiveOut;
};
#endif // REVNGC_LIVENESS_ANALYSIS_PASS_H
