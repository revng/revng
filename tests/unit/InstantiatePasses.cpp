/// \file InstantiatePasses.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE InstantiatePasses
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/IR/PassManager.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Model/SerializeModelPass.h"

template<typename T>
void instantiateFunctionAnalysis(llvm::Function *F) {
  llvm::FunctionPassManager FPM;
  llvm::FunctionAnalysisManager FAM;
  FAM.registerPass([]() { return T(); });
  FPM.run(*F, FAM);
}

template<typename T>
void instantiateModuleAnalysis(llvm::Module *M) {
  llvm::ModulePassManager MPM;
  llvm::ModuleAnalysisManager MAM;
  MAM.registerPass([]() { return T(); });
  MPM.run(*M, MAM);
}

template<typename T>
void instantiateModulePass(llvm::Module *M) {
  llvm::ModulePassManager MPM;
  llvm::ModuleAnalysisManager MAM;
  MPM.addPass(T());
  MPM.run(*M, MAM);
}

BOOST_AUTO_TEST_CASE(TestLoadModelAnalysis) {
  if (false) {
    instantiateFunctionAnalysis<LoadModelAnalysis>(nullptr);
    instantiateFunctionAnalysis<LoadModelAnalysis>(nullptr);
  }
}

BOOST_AUTO_TEST_CASE(TestSerializeModelPass) {
  if (false) {
    instantiateModulePass<SerializeModelPass>(nullptr);
  }
}
