#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/DOTGraphTraits.h"

#include "revng-c/RestructureCFG/CycleEquivalenceAnalysis.h"
#include "revng-c/RestructureCFG/CycleEquivalenceClass.h"

class CycleEquivalencePass : public llvm::FunctionPass {

private:
  // Resulting map containing, for each edge, the corresponding `Cycle
  // Equivalence Class` ID
  CycleEquivalenceAnalysis<llvm::Function *>::CycleEquivalenceResult
    EdgeToCycleEquivalenceClassIDMap;

public:
  static char ID;

public:
  CycleEquivalencePass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  const CycleEquivalenceAnalysis<llvm::Function *>::CycleEquivalenceResult &
  getResult();
};
