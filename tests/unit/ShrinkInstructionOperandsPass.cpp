/// \file ShrinkInstructionOperandsPass.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE ShrinkInstructionOperandsPass
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"

#include "revng/BasicAnalyses/ShrinkInstructionOperandsPass.h"
#include "revng/Support/IRHelpers.h"
#include "revng/UnitTestHelpers/LLVMTestHelpers.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

using namespace llvm;

static unsigned getSize(llvm::Value *V) {
  llvm::Type *T = V->getType();
  if (auto *IntTy = llvm::dyn_cast<llvm::IntegerType>(T))
    return IntTy->getBitWidth();

  return 0;
}

static Function *run(Module *M, const char *Body) {
  Function *F = M->getFunction("main");

  FunctionPassManager FPM;
  FPM.addPass(ShrinkInstructionOperandsPass());

  FunctionAnalysisManager FAM;

  ModuleAnalysisManager MAM;
  FAM.registerPass([&MAM] { return ModuleAnalysisManagerFunctionProxy(MAM); });

  PassBuilder PB;
  PB.registerFunctionAnalyses(FAM);
  PB.registerModuleAnalyses(MAM);

  FPM.run(*F, FAM);

  return F;
}

BOOST_AUTO_TEST_CASE(Comparison) {
  const char *Body = R"LLVM(
  %op1 = add i32 0, 0
  %op2 = add i32 0, 0
  %op1x = zext i32 %op1 to i64
  %op2x = zext i32 %op2 to i64
  %cmp = icmp ugt i64 %op1x, %op2x
  ret void
)LLVM";

  LLVMContext TestContext;
  std::unique_ptr<Module> M = loadModule(TestContext, Body);
  Function *F = run(M.get(), Body);

  auto *Cmp = instructionByName(F, "cmp");
  revng_check(getSize(Cmp->getOperand(0)) == 32);
  revng_check(getSize(Cmp->getOperand(1)) == 32);
}

BOOST_AUTO_TEST_CASE(DontCareBinary) {
  const char *Body = R"LLVM(
  %op1 = add i32 0, 0
  %op2 = add i32 0, 0
  %op1x = zext i32 %op1 to i64
  %op2x = zext i32 %op2 to i64
  %add = add i64 %op1x, %op2x
  %addt = trunc i64 %add to i32
  %final_use = add i32 %addt, 0
  ret void
)LLVM";

  LLVMContext TestContext;
  std::unique_ptr<Module> M = loadModule(TestContext, Body);
  Function *F = run(M.get(), Body);

  auto *Add = instructionByName(F, "add");

  revng_check(getSize(Add) == 32);
  revng_check(getSize(Add->getOperand(0)) == 32);
  revng_check(getSize(Add->getOperand(1)) == 32);
}
