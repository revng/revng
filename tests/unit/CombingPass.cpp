/// \file CombingPass.cpp
/// Tests for CombingPass

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdlib>

#define BOOST_TEST_MODULE CombingPass
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"

#include "revng/RestructureCFG/BasicBlockNode.h"
#include "revng/RestructureCFG/BasicBlockNodeImpl.h"
#include "revng/RestructureCFG/RegionCFGTree.h"
#include "revng/RestructureCFG/RegionCFGTreeImpl.h"
#include "revng/Support/Debug.h"
#include "revng/UnitTestHelpers/DotGraphObject.h"

using namespace llvm;

// Specialization of the `WeightTraits` for the `DotNode` class. In this
// situation we simply use 1 as default weight.
template<>
struct WeightTraits<DotNode *> {
  static inline size_t getWeight(DotNode *) { return 1; }
};

struct ArgsFixture {
  int argc;
  char **argv;

  ArgsFixture() :
    argc(boost::unit_test::framework::master_test_suite().argc),
    argv(boost::unit_test::framework::master_test_suite().argv) {}
};

enum TestType {
  Equal,
  NotEqual
};

static void runTest(TestType Type,
                    std::string &InputFileName,
                    std::string &ReferenceFileName) {

  // TODO: the `BBNodeToDotNodeMap`s are ignored, but they are needed by the
  //      `initialize` member function. Consider making the parameter optional.

  // Load the input graph and populate a new `RegionCFG` starting from it.
  DotGraph InputDot = DotGraph();
  InputDot.parseDotFromFile(InputFileName, "entry");
  RegionCFG<DotNode *> Input = RegionCFG<DotNode *>();

  Input.initialize(&InputDot);

  // Load the reference graph and populate a `RegionCFG` starting from it.
  DotGraph ReferenceDot = DotGraph();
  ReferenceDot.parseDotFromFile(ReferenceFileName, "entry");
  RegionCFG<DotNode *> Reference = RegionCFG<DotNode *>();

  Reference.initialize(&ReferenceDot);

  // Apply the combing pass to the input `RegionCFG`.
  Input.inflate();

  // Save the result of the comb pass.
  // Input.dumpCFGOnFile(DotPath + "output.dot");

  //
  // Check that the reference graph and the combed one are equivalent.
  //

  if (Type == Equal) {
    revng_assert(Input.isTopologicallyEquivalent(Reference));
  } else if (Type == NotEqual) {
    revng_assert(!Input.isTopologicallyEquivalent(Reference));
  } else {
    revng_abort("Test type not supported");
  }
}

BOOST_FIXTURE_TEST_SUITE(FixtureTestSuite, ArgsFixture)

BOOST_AUTO_TEST_CASE(TrivialGraphEqual) {
  std::string DotPath = argv[1];
  std::string InputFileName = DotPath + "trivial.dot";
  std::string ReferenceFileName = DotPath + "trivial.dot";

  runTest(Equal, InputFileName, ReferenceFileName);
}

BOOST_AUTO_TEST_CASE(SimpleGraphEqual) {
  std::string DotPath = argv[1];
  std::string InputFileName = DotPath + "simple.dot";
  std::string ReferenceFileName = DotPath + "simple.dot";

  runTest(Equal, InputFileName, ReferenceFileName);
}

BOOST_AUTO_TEST_CASE(SimpleGraphNotEqual) {
  std::string DotPath = argv[1];
  std::string InputFileName = DotPath + "simple.dot";
  std::string ReferenceFileName = DotPath + "trivial.dot";

  runTest(NotEqual, InputFileName, ReferenceFileName);
}

// End tag of test suite
BOOST_AUTO_TEST_SUITE_END()
