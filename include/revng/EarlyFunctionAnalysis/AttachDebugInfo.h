#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#include "revng/BasicAnalyses/CustomCFG.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Support/IRHelpers.h"

class AttachDebugInfo : public llvm::ModulePass {
public:
  static char ID;

public:
  AttachDebugInfo() : llvm::ModulePass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<FunctionMetadataCachePass>();
    AU.addRequired<GeneratedCodeBasicInfoWrapperPass>();
  }

  bool runOnModule(llvm::Module &M) override;
};
