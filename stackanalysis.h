#ifndef _STACKANALYSIS_H
#define _STACKANALYSIS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes

// LLVM includes
#include "llvm/Pass.h"

// Local includes
#include "generatedcodebasicinfo.h"

// Forward declarations
namespace llvm {
class BasicBlock;
}

class JumpTargetManager;

namespace StackAnalysis {

class StackAnalysis : public llvm::FunctionPass {
public:
  static char ID;

public:
  StackAnalysis() : llvm::FunctionPass(ID) { }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<GeneratedCodeBasicInfo>();
  }

  bool runOnFunction(llvm::Function &F) override;

  void serialize(std::ostream &Output) {
    Output << TextRepresentation;
  }

private:
  std::string TextRepresentation;
};

}

#endif // _STACKANALYSIS_H
