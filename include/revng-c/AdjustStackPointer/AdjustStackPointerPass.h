#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassPlugin.h"

struct AdjustStackPointerPass
  : public llvm::PassInfoMixin<AdjustStackPointerPass> {

public:
  llvm::PreservedAnalyses
  run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo();

struct LegacyPMAdjustStackPointerPass : public llvm::FunctionPass {
public:
  static char ID;

  LegacyPMAdjustStackPointerPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
};
