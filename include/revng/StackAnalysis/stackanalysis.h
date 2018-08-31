#ifndef _STACKANALYSIS_H
#define _STACKANALYSIS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <string>

// LLVM includes
#include "llvm/Pass.h"

// Local includes
#include "functioncallidentification.h"
#include "functionssummary.h"
#include "generatedcodebasicinfo.h"

namespace StackAnalysis {

template<bool AnalyzeABI>
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

  const std::set<const llvm::GlobalVariable *> &
  getClobbered(llvm::BasicBlock *Function) {
    return GrandResult.Functions[Function].ClobberedRegisters;
  }

  void serialize(std::ostream &Output) {
    Output << TextRepresentation;
  }

private:
  FunctionsSummary GrandResult;
  std::string TextRepresentation;

};

} // namespace StackAnalysis

#endif // _STACKANALYSIS_H
