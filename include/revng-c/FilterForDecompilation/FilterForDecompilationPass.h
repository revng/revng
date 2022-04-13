#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/Pass.h"

// Forward declarations
namespace llvm {

class Function;
class Module;

} // end namespace llvm

struct FilterForDecompilationFunctionPass : public llvm::FunctionPass {
public:
  static char ID;

  FilterForDecompilationFunctionPass() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};

struct FilterForDecompilationModulePass : public llvm::ModulePass {
public:
  static char ID;

  FilterForDecompilationModulePass() : llvm::ModulePass(ID) {}

  bool runOnModule(llvm::Module &F) override;

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;
};
