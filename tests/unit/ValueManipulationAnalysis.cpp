/// \file ValueManipulationAnalysis.cpp
/// Test the ValueManipulationAnalysis analysis

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE ValueManipulationAnalysis
bool init_unit_test();

#include "boost/test/unit_test.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Verifier.h"

#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/UnitTestHelpers/LLVMTestHelpers.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

#include "revng-c/ValueManipulationAnalysis/TypeColors.h"
#include "revng-c/ValueManipulationAnalysis/VMAPipeline.h"
#include "revng-c/ValueManipulationAnalysis/ValueManipulationAnalysis.h"

#include "lib/ValueManipulationAnalysis/Mincut.h"
#include "lib/ValueManipulationAnalysis/TypeFlowGraph.h"
#include "lib/ValueManipulationAnalysis/TypeFlowNode.h"

using namespace llvm;
using namespace vma;

// Node types
enum NodeType : unsigned {
  VALUE,
  USE,
  MAX_TYPES
};
const inline std::string NodeTypeName[MAX_TYPES] = { "values", "uses  " };

// Counters
using ColorCounter = std::array<unsigned, MAX_COLORS>;
using TypeCounter = std::array<unsigned, MAX_TYPES>;

struct ExpectedShape {
  TypeCounter Types;
  ColorCounter Colors;
  unsigned NCasts;
  unsigned NUndecided;

  ExpectedShape(TypeCounter T,
                ColorCounter C,
                unsigned NCasts,
                unsigned NUndecided) :
    Types(T), Colors(C), NCasts(NCasts), NUndecided(NUndecided) {}
};

/// Keep a count of how many nodes are colored with each color
static ColorCounter countColors(const TypeFlowGraph &TG) {
  ColorCounter CC = { 0 };

  for (const TypeFlowNode *const N : nodes(&TG))
    for (size_t I = 0; I < MAX_COLORS; I++)
      CC[I] += N->getCandidates().Bits.test(I);

  return CC;
}

/// Keep a count of how many nodes of each type there are in \a TG
static TypeCounter countTypes(const TypeFlowGraph &TG) {
  TypeCounter TC = { 0 };

  for (const TypeFlowNode *const N : nodes(&TG)) {
    if (N->isUse())
      TC[USE] += 1;
    else if (N->isValue())
      TC[VALUE] += 1;
    else
      revng_abort();
  }

  return TC;
}

/// Check that the TG nodes have the expected types and colors
static void checkShape(const TypeFlowGraph &TG, const ExpectedShape &Expected) {
  const TypeCounter &ExpectedTypes = Expected.Types;
  const ColorCounter &ExpectedColors = Expected.Colors;
  const TypeCounter ActualType = countTypes(TG);
  const ColorCounter ActualColor = countColors(TG);

  dbgs() << "========================\n";
  dbgs() << "What    Expected  Actual\n";

  // Check that there are the expected number of nodes of each type
  for (size_t I = 0; I < MAX_TYPES; I++) {
    dbgs() << NodeTypeName[I] << "\t" << ExpectedTypes[I] << "\t"
           << ActualType[I] << "\n";
    revng_check(ExpectedTypes[I] == ActualType[I]);
  }

  // Check that there are the expected number of nodes of each color
  for (size_t I = 0; I < MAX_COLORS; I++) {
    dbgs() << vma::TypeColorName[I] << "    \t" << ExpectedColors[I] << "\t"
           << ActualColor[I] << "\n";
    revng_check(ActualColor[I] == ExpectedColors[I]);
  }

  // Check number of casts in the graph
  unsigned ActualCasts = countCasts(TG);
  dbgs() << "casts \t" << Expected.NCasts << "\t" << ActualCasts << "\n";
  revng_check(Expected.NCasts == ActualCasts);

  // Check number of undecided nodes
  auto IsUndecided = [](const TypeFlowNode *N) { return N->isUndecided(); };
  unsigned NUndecided = llvm::count_if(TG.nodes(), IsUndecided);
  dbgs() << "undecided\t" << Expected.NUndecided << "\t" << NUndecided << "\n";
  revng_check(NUndecided == Expected.NUndecided);
}

/// Check that the information in the graph are consistent
static void checkTGCorrectness(TypeFlowGraph &TG) {
  // Check consistency between the graph and the reverse map
  for (TypeFlowNode *N : TG.nodes()) {
    auto MapIter = TG.ContentToNodeMap.find(N->getContent());
    revng_check(MapIter != TG.ContentToNodeMap.end());
    revng_check(MapIter->second == N);
  }

  for (auto &Elem : TG.ContentToNodeMap) {
    revng_check(Elem.second != nullptr);

    auto NodeIter = llvm::find(TG.nodes(), Elem.second);
    revng_check(NodeIter != TG.nodes().end());
    revng_check((*NodeIter)->getContent() == Elem.first);
  }

  for (const TypeFlowNode *N : TG.nodes()) {
    // Candidates should be a subset of accepted colors
    revng_check(N->getAccepted().contains(N->getCandidates()));
    // Nodes can contain either uses or values
    revng_check(N->isUse() xor N->isValue());

    // If a Use node is in the TG, there must also be a node associated to the
    // user and one for the used value
    if (N->isUse()) {
      revng_check(TG.ContentToNodeMap.contains(N->getUse()->get()));
      revng_check(TG.ContentToNodeMap.contains(N->getUse()->getUser()));
    }

    // No double edges between nodes
    for (const auto *Succ : N->successors())
      revng_check(llvm::count(N->successors(), Succ));
  }
}

/// Check that a TypeFlowGraph is initialized correctly from a function
static void checkInit(const char *Body,
                      const ExpectedShape ExpectedInit,
                      const ExpectedShape ExpectedAfterProp,
                      const ExpectedShape ExpectedFinal) {
  FunctionMetadataCache Cache;
  // Read the LLVM IR
  LLVMContext C;
  std::unique_ptr<llvm::Module> M = loadModule(C, Body);

  Function *F = M->getFunction("main");

  // Build the TG
  TypeFlowGraph TG = makeTypeFlowGraphFromFunction(Cache, F, /*Model=*/nullptr);
  LLVMInitializer Init;
  Init.initializeColors(&TG);

  checkTGCorrectness(TG);
  checkShape(TG, ExpectedInit);

  // Propagate
  propagateColors(TG);
  checkTGCorrectness(TG);
  checkShape(TG, ExpectedAfterProp);

  // Numberness
  propagateNumberness(TG);
  for (auto *N : TG.nodes())
    revng_check(not N->getCandidates().Bits.test(NUMBERNESS_INDEX));

  // Undirected graph
  makeBidirectional(TG);
  for (auto *N : TG.nodes())
    revng_check(llvm::size(N->successors()) == llvm::size(N->predecessors()));

  // Mincut
  minCut(TG);
  checkShape(TG, ExpectedFinal);
}

// Test cases

BOOST_AUTO_TEST_CASE(TestTGInit) {

  VerifyLog.enable();

  // Alloca
  checkInit(R"LLVM(
    %0 = alloca i32, align 4
    %1 = alloca i32, align 4
    unreachable
  )LLVM",
            /*Init=*/
            { { /*values=*/3,
                /*uses=*/0 },
              { /*pointers=*/2,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/0,
              /*undecided=*/0 },
            /*After propagation=*/
            { { /*values=*/3,
                /*uses=*/0 },
              { /*pointers=*/2,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/0,
              /*undecided=*/0 },
            /*Final=*/
            { { /*values=*/3,
                /*uses=*/0 },
              { /*pointers=*/2,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/0,
              /*undecided=*/0 });

  // Alloca + Load
  checkInit(R"LLVM(
    %0 = alloca i32, align 4
    %1 = alloca i32, align 4
    %2 = load i32, i32* %0, align 4
    %3 = load i32, i32* %1, align 4
    unreachable
  )LLVM",
            /*Init=*/
            { { /*values=*/5,
                /*uses=*/2 },
              { /*pointers=*/4,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/0,
              /*undecided=*/0 },
            /*After propagation=*/
            { { /*values=*/5,
                /*uses=*/2 },
              { /*pointers=*/4,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/0,
              /*undecided=*/0 },
            /*Final=*/
            { { /*values=*/5,
                /*uses=*/2 },
              { /*pointers=*/4,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/0,
              /*undecided=*/0 });

  // Ptr forward (with add)
  checkInit(R"LLVM(
    %0 = alloca i32, align 4
    %1 = alloca i32, align 4
    %2 = load i32, i32* %0, align 4
    %3 = load i32, i32* %1, align 4
    %4 = add i32 %2, %3
    %5 = sdiv i32 %2, 2
    %6 = inttoptr i32 %3 to i32*
    store i32 10, i32* %6, align 4
    unreachable
  )LLVM",
            /*Init=*/
            { { /*values=*/9,
                /*uses=*/7 },
              { /*pointers=*/5,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/2,
                /*floats=*/0,
                /*numbers=*/1 },
              /*casts=*/2,
              /*undecided=*/0 },
            /*After propagation=*/
            { { /*values=*/9,
                /*uses=*/7 },
              { /*pointers=*/10, //< pointerness doesn't go past add
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/9,
                /*floats=*/0,
                /*numbers=*/1 },
              /*casts=*/0,
              /*undecided=*/5 },
            /*Final=*/
            { { /*values=*/9,
                /*uses=*/7 },
              { /*pointers=*/10,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/4,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/1,
              /*undecided=*/0 });

  // Ptr backward (with add)
  checkInit(R"LLVM(
    %0 = alloca i32, align 4
    %1 = alloca i32, align 4
    %2 = load i32, i32* %0, align 4
    %3 = load i32, i32* %1, align 4
    %4 = add i32 %2, %3
    %5 = inttoptr i32 %4 to i32*
    store i32 10, i32* %5, align 4
    unreachable
  )LLVM",
            /*Init=*/
            { { /*value=*/8,
                /*uses=*/6 },
              { /*pointers=*/5,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/1,
              /*undecided=*/0 },
            /*After propagation=*/
            { { /*value=*/8,
                /*uses=*/6 },
              { /*pointers=*/8,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/2,
              /*undecided=*/0 },
            /*Final=*/
            { { /*value=*/8,
                /*uses=*/6 },
              { /*pointers=*/8,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/2,
              /*undecided=*/0 });

  // Side propagation (with add)
  checkInit(R"LLVM(
    %0 = alloca i32, align 4
    %1 = alloca i32, align 4
    %2 = load i32, i32* %0, align 4
    %3 = load i32, i32* %1, align 4
    %4 = add i32 %2, %3
    %5 = sdiv i32 %2, 2
    %6 = udiv i32 %3, 3
    unreachable
  )LLVM",
            /*Init=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/4,
                /*unsigned=*/2,
                /*bools=*/0,
                /*signed=*/2,
                /*floats=*/0,
                /*numbers=*/2 },
              /*casts=*/2,
              /*undecided=*/0 },
            /*After propagation=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/4,
                /*unsigned=*/7,
                /*bools=*/0,
                /*signed=*/7,
                /*floats=*/0,
                /*numbers=*/2 },
              /*casts=*/0,
              /*undecided=*/5 },
            /*Final=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/4,
                /*unsigned=*/7,
                /*bools=*/0,
                /*signed=*/2,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/1,
              /*undecided=*/0 });

  // Side propagation (with icmp)
  checkInit(R"LLVM(
    %0 = alloca i32, align 4
    %1 = alloca i32, align 4
    %2 = load i32, i32* %0, align 4
    %3 = load i32, i32* %1, align 4
    %4 = icmp eq i32 %2, %3
    %5 = icmp sgt i32 %2, 2
    %6 = icmp ugt i32 %3, 3
    unreachable
  )LLVM",
            /*Init=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/4,
                /*unsigned=*/1,
                /*bools=*/3,
                /*signed=*/1,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/2,
              /*undecided=*/0 },
            /*After propagation=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/4,
                /*unsigned=*/5,
                /*bools=*/3,
                /*signed=*/5,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/0,
              /*undecided=*/4 },
            /*Final=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/4,
                /*unsigned=*/5,
                /*bools=*/3,
                /*signed=*/1,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/1,
              /*undecided=*/0 });

  // Mul + ptr (backward)
  checkInit(R"LLVM(
    %0 = alloca i32, align 4
    %1 = alloca i32, align 4
    %2 = alloca i32, align 4
    %3 = load i32, i32* %0, align 4
    %4 = load i32, i32* %1, align 4
    %5 = load i32, i32* %2, align 4
    %6 = mul i32 %3, %4
    %7 = add i32 %5, %6
    %8 = inttoptr i32 %7 to i32*
    store i32 10, i32* %8
    unreachable
  )LLVM",
            /*Init=*/
            { { /*values=*/11,
                /*uses=*/9 },
              { /*pointers=*/7,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/1 },
              /*casts=*/1,
              /*undecided=*/0 },
            /*After propagation=*/
            { { /*values=*/11,
                /*uses=*/9 },
              { /*pointers=*/10,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/1 },
              /*casts=*/2,
              /*undecided=*/0 },
            /*Final=*/
            { { /*values=*/11,
                /*uses=*/9 },
              { /*pointers=*/10,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/2,
              /*undecided=*/0 });

  // Mul + ptr (forward)
  checkInit(R"LLVM(
    %0 = alloca i32, align 4
    %1 = alloca i32, align 4
    %2 = alloca i32, align 4
    %3 = load i32, i32* %0, align 4
    %4 = load i32, i32* %1, align 4
    %5 = load i32, i32* %2, align 4
    %6 = mul i32 %3, %4
    %7 = add i32 %5, %6
    %8 = inttoptr i32 %5 to i32*
    store i32 10, i32* %8
    unreachable
  )LLVM",
            /*Init=*/
            { { /*values=*/11,
                /*uses=*/9 },
              { /*pointers=*/7,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/1 },
              /*casts=*/1,
              /*undecided=*/0 },
            /*After propagation=*/
            { { /*values=*/11,
                /*uses=*/9 },
              { /*pointers=*/12,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/1 },
              /*casts=*/1,
              /*undecided=*/0 },
            /*Final=*/
            { { /*values=*/11,
                /*uses=*/9 },
              { /*pointers=*/12,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/1,
              /*undecided=*/0 });

  // And with bools
  checkInit(R"LLVM(
    %0 = alloca i32, align 4
    %1 = alloca i32, align 4
    %2 = load i32, i32* %0, align 4
    %3 = load i32, i32* %1, align 4
    %4 = icmp eq i32 %2, 2
    %5 = icmp eq i32 %3, 3
    %6 = and i1 %5, %4
    unreachable
  )LLVM",
            /*Init=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/4,
                /*unsigned=*/0,
                /*bools=*/2,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/1 },
              /*casts=*/2,
              /*undecided=*/0 },
            /*After propagation=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/4,
                /*unsigned=*/0,
                /*bools=*/5,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/1 },
              /*casts=*/0,
              /*undecided=*/0 },
            /*Final=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/4,
                /*unsigned=*/0,
                /*bools=*/5,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/0,
              /*undecided=*/0 });

  // Shift ptr
  checkInit(R"LLVM(
    %0 = alloca i32, align 4
    %1 = alloca i32, align 4
    %2 = load i32, i32* %0, align 4
    %3 = load i32, i32* %1, align 4
    %4 = inttoptr i32 %2 to i32*
    store i32 10, i32* %4
    %5 = shl i32 %2, %3
    unreachable
  )LLVM",
            /*Init=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/5,
                /*unsigned=*/1,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/2,
              /*undecided=*/0 },
            /*After propagation=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/8,
                /*unsigned=*/2,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/1,
              /*undecided=*/0 },
            /*Final=*/
            { { /*values=*/8,
                /*uses=*/6 },
              { /*pointers=*/8,
                /*unsigned=*/2,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/1,
              /*undecided=*/0 });

  // Set last bit to 0
  checkInit(R"LLVM(
    %0 = alloca i32, align 4
    %1 = load i32, i32* %0, align 4
    %2 = inttoptr i32 %1 to i32*
    store i32 10, i32* %2
    %3 = and i32 %1, 254
    unreachable
  )LLVM",
            /*Init=*/
            { { /*values=*/6,
                /*uses=*/4 },
              { /*pointers=*/3,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/1 },
              /*casts=*/1,
              /*undecided=*/0 },
            /*Final=*/
            { { /*values=*/6,
                /*uses=*/4 },
              { /*pointers=*/6,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/1 },
              /*casts=*/1,
              /*undecided=*/0 },
            /*After propagation=*/
            { { /*values=*/6,
                /*uses=*/4 },
              { /*pointers=*/6,
                /*unsigned=*/0,
                /*bools=*/0,
                /*signed=*/0,
                /*floats=*/0,
                /*numbers=*/0 },
              /*casts=*/1,
              /*undecided=*/0 });
}

BOOST_AUTO_TEST_CASE(TestMajorityVoting) {
  const auto AddNode = [](TypeFlowGraph &G, unsigned Color) {
    UseOrValue DummyContent;
    ColorSet DecidedColor(Color);
    NodeColorProperty InitColors(DecidedColor, DecidedColor);

    return G.addNode(DummyContent, InitColors);
  };

  TypeFlowGraph G;
  TypeFlowNode *Undecided1 = AddNode(G, (POINTERNESS | UNSIGNEDNESS));
  TypeFlowNode *Undecided2 = AddNode(G, (POINTERNESS | UNSIGNEDNESS));
  Undecided1->addSuccessor(Undecided2);

  TypeFlowNode *Decided1 = AddNode(G, POINTERNESS);
  TypeFlowNode *Decided2 = AddNode(G, POINTERNESS);
  TypeFlowNode *Decided3 = AddNode(G, UNSIGNEDNESS);
  TypeFlowNode *Decided4 = AddNode(G, SIGNEDNESS);
  Undecided1->addSuccessor(Decided1);
  Undecided1->addSuccessor(Decided2);
  Undecided1->addSuccessor(Decided3);
  Undecided1->addSuccessor(Decided4);

  makeBidirectional(G);
  applyMajorityVoting(G);

  revng_check(Undecided1->getCandidates() == (POINTERNESS | UNSIGNEDNESS));
  revng_check(Undecided2->getCandidates() == (POINTERNESS | UNSIGNEDNESS));

  TypeFlowNode *Decided5 = AddNode(G, POINTERNESS);
  Undecided1->addSuccessor(Decided5);

  makeBidirectional(G);
  applyMajorityVoting(G);

  revng_check(Undecided1->getCandidates() == POINTERNESS);
  revng_check(Undecided2->getCandidates() == POINTERNESS);
}
