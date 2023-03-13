/// \file IRHelpers.cpp
/// \brief Tests for IRHelpers

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE IRHelpers
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/IR/DebugInfoMetadata.h"

#include "revng/Support/IRHelpers.h"
#include "revng/UnitTestHelpers/LLVMTestHelpers.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

using namespace llvm;

const char *VisitorTestBody = R"LLVM(
  %a = add i64 0, 0
  br i1 true, label %first_if_true, label %first_if_false

first_if_true:
  %b = add i64 0, 0
  br label %center

first_if_false:
  %c = add i64 0, 0
  br label %center

center:
  %d = add i64 0, 0
  %target = add i64 0, 0
  br i1 true, label %second_if_true, label %second_if_false

second_if_true:
  %e = add i64 0, 0
  br label %end

second_if_false:
  %f = add i64 0, 0
  br label %end

end:
  %g = add i64 0, 0
  ret void
)LLVM";

BOOST_AUTO_TEST_CASE(TestBackwardBFSVisitor) {

  struct Visitor : public BackwardBFSVisitor<Visitor> {
    std::vector<std::string> VisitLog;

    VisitAction visit(instruction_range Range) {
      for (Instruction &I : Range)
        VisitLog.push_back(getName(&I));
      return Continue;
    }
  };

  LLVMContext TestContext;
  std::unique_ptr<Module> M = loadModule(TestContext, VisitorTestBody);
  Function *F = M->getFunction("main");

  Visitor V;
  V.run(instructionByName(F, "target"));

  const std::vector<std::string> GroundTruth = { "target",           "d",
                                                 "first_if_false:2", "c",
                                                 "first_if_true:2",  "b",
                                                 "initial_block:2",  "a" };
  revng_check(V.VisitLog == GroundTruth);
}

BOOST_AUTO_TEST_CASE(TestForwardBFSVisitor) {

  struct Visitor : public ForwardBFSVisitor<Visitor> {
    std::vector<std::string> VisitLog;

    VisitAction visit(instruction_range Range) {
      for (Instruction &I : Range)
        VisitLog.push_back(getName(&I));
      return Continue;
    }
  };

  LLVMContext TestContext;
  std::unique_ptr<Module> M = loadModule(TestContext, VisitorTestBody);
  Function *F = M->getFunction("main");

  Visitor V;
  V.run(instructionByName(F, "target"));

  const std::vector<std::string> GroundTruth = {
    "center:3", "e", "second_if_true:2", "f", "second_if_false:2", "g", "end:2"
  };
  revng_check(V.VisitLog == GroundTruth);
}

BOOST_AUTO_TEST_CASE(PruneDICompileUnits) {
  LLVMContext Context;
  Module M("", Context);

  auto GetFile = [&](StringRef FileName) {
    return DIFile::getDistinct(Context, FileName, "/path/to/dir");
  };

  auto GetTuple = [&]() { return MDTuple::getDistinct(Context, None); };

  auto CreateDICU = [&](StringRef FileName) {
    const auto Default = DICompileUnit::DebugNameTableKind::Default;
    return DICompileUnit::getDistinct(Context,
                                      1,
                                      GetFile(FileName),
                                      "clang",
                                      false,
                                      "-g",
                                      2,
                                      "",
                                      DICompileUnit::FullDebug,
                                      GetTuple(),
                                      GetTuple(),
                                      GetTuple(),
                                      GetTuple(),
                                      GetTuple(),
                                      0,
                                      true,
                                      false,
                                      Default,
                                      false,
                                      "/",
                                      "");
  };

  // Create two compile units
  auto *UnusedDICU = CreateDICU("unused.c");
  auto *ActiveDICU = CreateDICU("active.c");

  // Record the compile units in llvm.dbg.cu
  auto *NamedMDNode = M.getOrInsertNamedMetadata("llvm.dbg.cu");
  NamedMDNode->addOperand(UnusedDICU);
  NamedMDNode->addOperand(ActiveDICU);

  // Create a function with a single instruction using active.c
  Type *VoidTy = Type::getVoidTy(Context);
  auto *FTy = FunctionType::get(VoidTy, false);
  auto *F = Function::Create(FTy, GlobalValue::ExternalLinkage, 0, "", &M);
  auto *BB = BasicBlock::Create(Context, "", F);
  auto *I = ReturnInst::Create(Context, BB);
  auto *Location = DILocation::get(Context, 0, 0, ActiveDICU);
  I->setDebugLoc({ Location });

  // Perform pruning and ensure it has been effective
  revng_check(NamedMDNode->getNumOperands() == 2);

  pruneDICompileUnits(M);

  revng_check(NamedMDNode->getNumOperands() == 1);
}
