/// \file IRHelpers.cpp
/// \brief Tests for IRHelpers

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Boost includes
#define BOOST_TEST_MODULE IRHelpers
bool init_unit_test();
#include <boost/test/unit_test.hpp>

// Local libraries includes
#include "revng/Support/IRHelpers.h"

// Local includes
#include "LLVMTestHelpers.h"

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

  const std::vector<std::string> GroundTruth = {
    "d", "first_if_false:2", "c", "first_if_true:2", "b", "initial_block:2", "a"
  };
  revng_assert(V.VisitLog == GroundTruth);
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
  revng_assert(V.VisitLog == GroundTruth);
}
