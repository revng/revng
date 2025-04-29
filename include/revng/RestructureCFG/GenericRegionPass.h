#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

#include "revng/RestructureCFG/GenericRegionInfo.h"
#include "revng/RestructureCFG/ScopeGraphGraphTraits.h"

class GenericRegionPass : public llvm::FunctionPass {
private:
  GenericRegionInfo<Scope<llvm::Function *>> GRI;

public:
  static char ID;

public:
  GenericRegionPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  const GenericRegionInfo<Scope<llvm::Function *>> &getResult();
};
