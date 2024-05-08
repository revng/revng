/// \file IRHelpers.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#define BOOST_TEST_MODULE IRHelpers
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/Utils/Cloning.h"

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

BOOST_AUTO_TEST_CASE(UniqueString) {
  LLVMContext Context;
  auto M = std::make_unique<Module>("", Context);
  const char *Namespace = "test.";
  const char *String1 = "string1";

  auto VariableExists = [&M, Namespace](StringRef Name) -> bool {
    std::string GlobalName = (Twine(Namespace) + Twine(Name)).str();
    return M->getGlobalVariable(GlobalName) != nullptr;
  };

  Constant *String1Constant1 = getUniqueString(&*M, String1, Namespace);

  // Test a global with the expected name has been created
  revng_check(VariableExists(String1));

  Constant *String1Constant2 = getUniqueString(&*M, String1, Namespace);

  // Check that the two strings are the same object
  revng_check(String1Constant1 == String1Constant2);

  const char *String2 = "string2";
  Constant *String2Constant = getUniqueString(&*M, String2, Namespace);
  revng_check(VariableExists(String2));

  // Check that the two strings are not the same object
  revng_check(String1Constant1 != String2Constant);

  // Test a string containing spaces
  const char *StringWithSpaces = "This contains spaces";
  getUniqueString(&*M, StringWithSpaces, Namespace);
  revng_check(not VariableExists(StringWithSpaces));

  // Test a string containing non-printable characters
  const char *NonPrintable = "NonPrintableChar\x01Here";
  getUniqueString(&*M, NonPrintable, Namespace);
  revng_check(not VariableExists(NonPrintable));

  // Test a long string
  const char *LongString = "ThisStringHasToBeLongerThanAHexEncodedSHA1Hash";
  revng_assert(StringRef(LongString).size() > 40);
  getUniqueString(&*M, LongString, Namespace);
  revng_check(not VariableExists(LongString));
}

BOOST_AUTO_TEST_CASE(PruneDICompileUnits) {
  LLVMContext Context;
  Module M("", Context);

  auto GetFile = [&](StringRef FileName) {
    return DIFile::getDistinct(Context, FileName, "/path/to/dir");
  };

  auto GetTuple = [&]() { return MDTuple::getDistinct(Context, {}); };

  auto CreateDICU = [&](DIFile *File) {
    const auto Default = DICompileUnit::DebugNameTableKind::Default;
    return DICompileUnit::getDistinct(Context,
                                      1,
                                      File,
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
  auto *UnusedDICU = CreateDICU(GetFile("unused.c"));
  auto *ActiveFile = GetFile("active.c");
  auto *ActiveDICU = CreateDICU(ActiveFile);

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
  auto *Subprogram = DISubprogram::get(Context,
                                       ActiveDICU,
                                       "",
                                       "",
                                       ActiveFile,
                                       0,
                                       nullptr,
                                       0,
                                       0,
                                       0,
                                       0,
                                       static_cast<DINode::DIFlags>(0),
                                       static_cast<DISubprogram::DISPFlags>(0),
                                       ActiveDICU,
                                       {},
                                       nullptr,
                                       {},
                                       {},
                                       {});
  auto *Location = DILocation::get(Context, 0, 0, Subprogram);
  I->setDebugLoc({ Location });

  // Perform pruning and ensure it has been effective
  revng_check(NamedMDNode->getNumOperands() == 2);

  pruneDICompileUnits(M);

  revng_check(NamedMDNode->getNumOperands() == 1);
}
