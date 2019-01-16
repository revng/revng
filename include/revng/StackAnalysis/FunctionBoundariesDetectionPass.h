#ifndef FUNCTIONBOUNDARIESDETECTIONPASS_H
#define FUNCTIONBOUNDARIESDETECTIONPASS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/Pass.h"

// Local libraries includes
#include "revng/StackAnalysis/StackAnalysis.h"

namespace StackAnalysis {

class FunctionBoundariesDetectionPass : public llvm::ModulePass {
public:
  static char ID;

public:
  FunctionBoundariesDetectionPass() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<StackAnalysis<false>>();
  }

  void serialize(std::ostream &Output, llvm::Module &M);

  bool runOnModule(llvm::Module &M) override;
};

} // namespace StackAnalysis

#endif // FUNCTIONBOUNDARIESDETECTIONPASS_H
