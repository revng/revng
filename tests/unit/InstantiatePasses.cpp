/// \file InstantiatePasses.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE InstantiatePasses
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/IR/PassManager.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

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

BOOST_AUTO_TEST_CASE(MaybeDrop) {
  BOOST_TEST(true);
}
