#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"

class VariableScopeAnalysisPass : public llvm::FunctionPass {
public:
  using ValuePtrSet = llvm::SmallPtrSet<const llvm::Value *, 32>;

public:
  static char ID;

public:
  VariableScopeAnalysisPass() :
    llvm::FunctionPass(ID), NeedsLoopStateVar(), TopScopeVariables() {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool needsLoopStateVar() const { return NeedsLoopStateVar; }

  const ValuePtrSet &getTopScopeVariables() const { return TopScopeVariables; }

private:
  bool NeedsLoopStateVar;
  ValuePtrSet TopScopeVariables;
};
