#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Passes/PassBuilder.h"

namespace detail {

class SimplePassManagerBase {
protected:
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::PassBuilder PB;

  SimplePassManagerBase() {
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  }
};

} // namespace detail

/// Wrapper for the new llvm pass manager. Does all the required initialization
/// and exposes all the methods to have an easy to use pass manager.
class SimplePassManager : public detail::SimplePassManagerBase {
private:
  llvm::ModulePassManager PM;

public:
  template<typename T>
  void addPass(T &&Pass) {
    PM.addPass(std::forward<T>(Pass));
  }

  void run(llvm::Module &Module) { PM.run(Module, MAM); }
};

class SimpleFunctionPassManager : public detail::SimplePassManagerBase {
private:
  llvm::FunctionPassManager PM;

public:
  template<typename T>
  void addPass(T &&Pass) {
    PM.addPass(std::forward<T>(Pass));
  }

  void run(llvm::Function &Function) { PM.run(Function, FAM); }
};
