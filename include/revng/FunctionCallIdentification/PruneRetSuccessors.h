#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

// LLVM includes
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

// Local libraries includes
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/FunctionCallIdentification/FunctionCallIdentification.h"

class PruneRetSuccessors : public llvm::ModulePass {
public:
  static char ID;

public:
  PruneRetSuccessors() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
    AU.addRequired<FunctionCallIdentification>();
  }

  bool runOnModule(llvm::Module &M) override;
};
