/// \file FilteredGraphTraits.cpp
/// Tests for the FilteredGraphTraits template.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE FilteredGraphTraits
bool init_unit_test();
#include <functional>
#include <map>
#include <type_traits>

#include "boost/test/unit_test.hpp"

#include "llvm/ADT/BreadthFirstIterator.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/GenericDomTreeConstruction.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/UnitTestHelpers/DotGraphObject.h"
#include "revng/UnitTestHelpers/LLVMTestHelpers.h"
#include "revng/UnitTestHelpers/UnitTestHelpers.h"

using namespace llvm;

template<char Letter>
static bool
hasLetterInName(const llvm::GraphTraits<Function *>::NodeRef & /*Src*/,
                const llvm::GraphTraits<Function *>::NodeRef &Tgt) {
  if (not Tgt->hasName())
    return false;
  return Tgt->getName().contains(Letter);
}

template<char Letter>
static bool
edgeHasLetterInName(const llvm::GraphTraits<DotGraph *>::EdgeRef &E) {
  return E.second->getName().contains(Letter);
}

template<typename T>
static bool alwaysTrue(const T & /*Src*/, const T & /*Tgt*/) {
  return true;
}

template<typename NodeT>
static bool alwaysTrueEdge(const typename GraphTraits<NodeT>::EdgeRef &) {
  return true;
}

template<typename T>
static bool alwaysFalse(const T & /*Src*/, const T & /*Tgt*/) {
  return false;
}

template<typename NodeT>
static bool alwaysFalseEdge(const typename GraphTraits<NodeT>::EdgeRef &) {
  return false;
}

template<typename NodeT>
using TrueNPFG = NodePairFilteredGraph<NodeT, alwaysTrue<NodeT>>;

template<typename NodeT>
using FalseNPFG = NodePairFilteredGraph<NodeT, alwaysFalse<NodeT>>;

template<typename NodeT>
using TrueEFG = EdgeFilteredGraph<NodeT, alwaysTrueEdge<NodeT>>;

template<typename NodeT>
using FalseEFG = EdgeFilteredGraph<NodeT, alwaysFalseEdge<NodeT>>;

struct ArgsFixture {
  int ArgC;
  char **ArgV;

  ArgsFixture() :
    ArgC(boost::unit_test::framework::master_test_suite().argc),
    ArgV(boost::unit_test::framework::master_test_suite().argv) {}

  ~ArgsFixture() {}
};

BOOST_FIXTURE_TEST_SUITE(TestSuite, ArgsFixture)

BOOST_AUTO_TEST_CASE(NodePairFilteredGraphTests) {

  std::string Body{ R"LLVM(
  br label %starter

starter:
  %to_store = load i64, i64* @rax
  %gt10 = icmp ugt i64 %to_store, 10
  %lt15 = icmp ult i64 %to_store, 15
  %in10_15 = and i1 %gt10, %lt15
  br i1 %in10_15, label %end, label %false

false:
  %gt30 = icmp ugt i64 %to_store, 30
  %lt35 = icmp ult i64 %to_store, 35
  %in30_35 = and i1 %gt30, %lt35
  br i1 %in30_35, label %end, label %_xit

_xit:
  unreachable

end:
  store i64 %to_store, i64* @pc
  unreachable

)LLVM" };

  LLVMContext C;
  std::unique_ptr<llvm::Module> M = loadModule(C, Body.data());

  Function *F = M->getFunction("main");
  BasicBlock *BackBB = &F->back();

  // #### Depth first visits ####

  std::vector<BasicBlock *> DefaultResults;
  std::vector<llvm::StringRef> DefaultResultsNames{
    "initial_block", "starter", "end", "false", "_xit",
  };
  { // Baseline, on unfiltered graph
    for (auto &It : llvm::depth_first(F))
      DefaultResults.push_back(It);
    revng_check(DefaultResults.size() == DefaultResultsNames.size());
    for (size_t I = 0; I < DefaultResults.size(); ++I)
      revng_check(DefaultResults[I]->getName() == DefaultResultsNames[I]);
  }

  { // Do-nothing filter
    std::vector<llvm::StringRef> ExpectedResultsNames{
      "initial_block", "starter", "end", "false", "_xit",
    };
    std::vector<BasicBlock *> Results;
    using FNPFG = NodePairFilteredGraph<Function *, alwaysTrue<BasicBlock *>>;
    for (auto &It : llvm::depth_first(FNPFG(F)))
      Results.push_back(It);
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I]->getName() == ExpectedResultsNames[I]);
  }
  { // Filter all, leaving no edge
    std::vector<llvm::StringRef> ExpectedResultsNames{
      "initial_block",
    };
    std::vector<BasicBlock *> Results;
    using NPFG = NodePairFilteredGraph<Function *, alwaysFalse<BasicBlock *>>;
    for (auto &It : llvm::depth_first(NPFG(F)))
      Results.push_back(It);
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I]->getName() == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards BB with name without 'a's
    std::vector<llvm::StringRef> ExpectedResultsNames = {
      "initial_block",
      "starter",
      "false",
    };
    std::vector<BasicBlock *> Results;
    using NPFG = NodePairFilteredGraph<Function *, hasLetterInName<'a'>>;
    for (auto &It : llvm::depth_first(NPFG(F)))
      Results.push_back(It);
    // Only the initial_block, starter, and false should be left
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I]->getName() == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards BB with name without 'e's
    // All but _xit should reached, in this order
    std::vector<llvm::StringRef> ExpectedResultsNames = {
      "initial_block",
      "starter",
      "end",
      "false",
    };
    std::vector<BasicBlock *> Results;
    using NPFG = NodePairFilteredGraph<Function *, hasLetterInName<'e'>>;
    for (auto &It : llvm::depth_first(NPFG(F)))
      Results.push_back(It);
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I]->getName() == ExpectedResultsNames[I]);
  }

  // #### Breadth first visits ####

  { // Baseline, on unfiltered graph
    std::vector<llvm::StringRef> ExpectedResultsNames{
      "initial_block", "starter", "end", "false", "_xit",
    };
    std::vector<BasicBlock *> Results;
    for (auto &It : llvm::breadth_first(F))
      Results.push_back(It);
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I]->getName() == ExpectedResultsNames[I]);
  }
  { // Do-nothing filter
    std::vector<llvm::StringRef> ExpectedResultsNames{
      "initial_block", "starter", "end", "false", "_xit",
    };
    std::vector<BasicBlock *> Results;
    using NPFG = NodePairFilteredGraph<Function *, alwaysTrue<BasicBlock *>>;
    for (auto &It : llvm::breadth_first(NPFG(F)))
      Results.push_back(It);
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I]->getName() == ExpectedResultsNames[I]);
  }
  { // Filter all, leaving no edge
    std::vector<llvm::StringRef> ExpectedResultsNames{
      "initial_block",
    };
    std::vector<BasicBlock *> Results;
    using NPFG = NodePairFilteredGraph<Function *, alwaysFalse<BasicBlock *>>;
    for (auto &It : llvm::breadth_first(NPFG(F)))
      Results.push_back(It);
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I]->getName() == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards BB with name without 'a's
    std::vector<BasicBlock *> Results;
    using NPFG = NodePairFilteredGraph<Function *, hasLetterInName<'a'>>;
    for (auto &It : llvm::breadth_first(NPFG(F)))
      Results.push_back(It);
    // Only the initial_block, starter, and false should be left
    std::vector<llvm::StringRef> ExpectedResultsNames = {
      "initial_block",
      "starter",
      "false",
    };
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I]->getName() == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards BB with name without 'e's
    std::vector<llvm::StringRef> ExpectedResultsNames = {
      "initial_block",
      "starter",
      "end",
      "false",
    };
    std::vector<BasicBlock *> Results;
    using NPFG = NodePairFilteredGraph<Function *, hasLetterInName<'e'>>;
    for (auto &It : llvm::breadth_first(NPFG(F)))
      Results.push_back(It);
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I]->getName() == ExpectedResultsNames[I]);
  }

  // #### Inverse depth first visits ####
  {
    std::vector<llvm::StringRef> ExpectedResultsNames{
      "end",
      "false",
      "starter",
      "initial_block",
    };
    std::vector<BasicBlock *> Results;
    for (auto &It : llvm::inverse_depth_first(BackBB))
      Results.push_back(It);
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I]->getName() == ExpectedResultsNames[I]);
  }
  { // Do-nothing filter
    std::vector<llvm::StringRef> ExpectedResultsNames{
      "end",
      "false",
      "starter",
      "initial_block",
    };
    std::vector<BasicBlock *> FirstResults, SecondResults;
    using NPFG = NodePairFilteredGraph<BasicBlock *, alwaysTrue<BasicBlock *>>;
    using InvBBNPFG = Inverse<NPFG>;
    for (auto &It : llvm::depth_first(InvBBNPFG(BackBB)))
      FirstResults.push_back(It);
    revng_check(FirstResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < FirstResults.size(); ++I)
      revng_check(FirstResults[I]->getName() == ExpectedResultsNames[I]);

    for (auto &It : llvm::inverse_depth_first(NPFG(BackBB)))
      SecondResults.push_back(It);
    revng_check(SecondResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < SecondResults.size(); ++I)
      revng_check(SecondResults[I]->getName() == ExpectedResultsNames[I]);
  }
  { // Filter all, leaving no edge
    std::vector<llvm::StringRef> ExpectedResultsNames{
      "end",
    };
    std::vector<BasicBlock *> FirstResults, SecondResults;
    using NPFG = NodePairFilteredGraph<BasicBlock *, alwaysFalse<BasicBlock *>>;
    using InvBBNPFG = Inverse<NPFG>;
    for (auto &It : llvm::depth_first(InvBBNPFG(BackBB)))
      FirstResults.push_back(It);
    revng_check(FirstResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < FirstResults.size(); ++I)
      revng_check(FirstResults[I]->getName() == ExpectedResultsNames[I]);

    for (auto &It : llvm::inverse_depth_first(NPFG(BackBB)))
      SecondResults.push_back(It);
    revng_check(SecondResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < SecondResults.size(); ++I)
      revng_check(SecondResults[I]->getName() == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards BB with name without 'a's
    std::vector<llvm::StringRef> ExpectedResultsNames{
      "end",
    };
    std::vector<BasicBlock *> FirstResults, SecondResults;
    using NPFG = NodePairFilteredGraph<BasicBlock *, hasLetterInName<'a'>>;
    using InvBBNPFG = Inverse<NPFG>;
    for (auto &It : llvm::depth_first(InvBBNPFG(BackBB)))
      FirstResults.push_back(It);
    revng_check(FirstResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < FirstResults.size(); ++I)
      revng_check(FirstResults[I]->getName() == ExpectedResultsNames[I]);

    for (auto &It : llvm::inverse_depth_first(NPFG(BackBB)))
      SecondResults.push_back(It);
    revng_check(SecondResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < SecondResults.size(); ++I)
      revng_check(SecondResults[I]->getName() == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards BB with name without 'e's
    std::vector<llvm::StringRef> ExpectedResultsNames{
      "end",
      "false",
      "starter",
      "initial_block",
    };
    std::vector<BasicBlock *> FirstResults, SecondResults;
    using NPFG = NodePairFilteredGraph<BasicBlock *, hasLetterInName<'e'>>;
    using InvBBNPFG = Inverse<NPFG>;
    for (auto &It : llvm::depth_first(InvBBNPFG(BackBB)))
      FirstResults.push_back(It);
    revng_check(FirstResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < FirstResults.size(); ++I)
      revng_check(FirstResults[I]->getName() == ExpectedResultsNames[I]);

    for (auto &It : llvm::inverse_depth_first(NPFG(BackBB)))
      SecondResults.push_back(It);
    revng_check(SecondResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < SecondResults.size(); ++I)
      revng_check(SecondResults[I]->getName() == ExpectedResultsNames[I]);
  }

  // #### Dominator and PostDominator Trees ####
  std::map<llvm::StringRef, BasicBlock *> NamesToBlocks;
  revng_check(DefaultResults.size() == DefaultResultsNames.size());
  for (size_t I = 0ULL; I < DefaultResults.size(); ++I)
    NamesToBlocks[DefaultResultsNames[I]] = DefaultResults[I];
  const auto Dominates = [&NamesToBlocks](auto &DT,
                                          const std::string &A,
                                          const std::string &B) -> bool {
    return DT.dominates(NamesToBlocks.at(A), NamesToBlocks.at(B));
  };
  {
    using DT = DominatorTreeBase<BasicBlock, false>;
    DT TheDomTree;
    TheDomTree.recalculate(*F);
    revng_check(TheDomTree.verify());
    TheDomTree.print(errs());
    revng_check(Dominates(TheDomTree, "initial_block", "starter"));
    revng_check(Dominates(TheDomTree, "starter", "end"));
    revng_check(Dominates(TheDomTree, "starter", "false"));
    revng_check(Dominates(TheDomTree, "false", "_xit"));
  }
  {
    using PostDT = DominatorTreeBase<BasicBlock, true>;
    PostDT ThePostDomTree;
    ThePostDomTree.recalculate(*F);
    revng_check(ThePostDomTree.verify());
    ThePostDomTree.print(errs());
    const auto *Sink = ThePostDomTree.getRootNode();
    const auto DTNodeDominates =
      [&NamesToBlocks](auto &DT,
                       const DomTreeNodeBase<BasicBlock> *A,
                       const std::string &B) -> bool {
      return DT.dominates(A, DT.getNode(NamesToBlocks.at(B)));
    };
    revng_check(DTNodeDominates(ThePostDomTree, Sink, "initial_block"));
    revng_check(DTNodeDominates(ThePostDomTree, Sink, "starter"));
    revng_check(DTNodeDominates(ThePostDomTree, Sink, "false"));
    revng_check(DTNodeDominates(ThePostDomTree, Sink, "end"));
    revng_check(DTNodeDominates(ThePostDomTree, Sink, "_xit"));
    revng_check(Dominates(ThePostDomTree, "starter", "initial_block"));
  }
  {
    using FiltDT = DominatorTreeOnView<BasicBlock, false, TrueNPFG>;
    FiltDT TheFiltDomTree;
    TheFiltDomTree.recalculate(*F);
    revng_check(TheFiltDomTree.verify());
    TheFiltDomTree.print(errs());
    revng_check(Dominates(TheFiltDomTree, "initial_block", "starter"));
    revng_check(Dominates(TheFiltDomTree, "starter", "end"));
    revng_check(Dominates(TheFiltDomTree, "starter", "false"));
    revng_check(Dominates(TheFiltDomTree, "false", "_xit"));
  }
  {
    using FiltDT = DominatorTreeOnView<BasicBlock, false, FalseNPFG>;
    FiltDT TheFiltDomTree;
    TheFiltDomTree.recalculate(*F);
    revng_check(TheFiltDomTree.verify());
    TheFiltDomTree.print(errs());
    revng_check(nullptr == TheFiltDomTree.getNode(NamesToBlocks.at("starter")));
    revng_check(nullptr == TheFiltDomTree.getNode(NamesToBlocks.at("false")));
    revng_check(nullptr == TheFiltDomTree.getNode(NamesToBlocks.at("end")));
    revng_check(nullptr == TheFiltDomTree.getNode(NamesToBlocks.at("_xit")));
    revng_check(TheFiltDomTree.getNode(NamesToBlocks.at("initial_block")));
    revng_check(Dominates(TheFiltDomTree, "initial_block", "initial_block"));
  }
  {
    using FiltDT = DominatorTreeOnView<BasicBlock, true, TrueNPFG>;
    FiltDT TheFiltDomTree;
    TheFiltDomTree.recalculate(*F);
    revng_check(TheFiltDomTree.verify());
    TheFiltDomTree.print(errs());
    const auto *Sink = TheFiltDomTree.getRootNode();
    const auto DTNodeDominates =
      [&NamesToBlocks](auto &DT,
                       const DomTreeNodeOnView<BasicBlock, TrueNPFG> *A,
                       const std::string &B) -> bool {
      return DT.dominates(A, DT.getNode(NamesToBlocks.at(B)));
    };
    revng_check(DTNodeDominates(TheFiltDomTree, Sink, "initial_block"));
    revng_check(DTNodeDominates(TheFiltDomTree, Sink, "starter"));
    revng_check(DTNodeDominates(TheFiltDomTree, Sink, "false"));
    revng_check(DTNodeDominates(TheFiltDomTree, Sink, "end"));
    revng_check(DTNodeDominates(TheFiltDomTree, Sink, "_xit"));
    revng_check(Dominates(TheFiltDomTree, "starter", "initial_block"));
  }
  {
    using FiltDT = DominatorTreeOnView<BasicBlock, true, FalseNPFG>;
    FiltDT TheFiltDomTree;
    TheFiltDomTree.recalculate(*F);
    revng_check(TheFiltDomTree.verify());
    TheFiltDomTree.print(errs());
    const auto *Sink = TheFiltDomTree.getRootNode();
    const auto DTNodeDominates =
      [&NamesToBlocks](auto &DT,
                       const DomTreeNodeOnView<BasicBlock, FalseNPFG> *A,
                       const std::string &B) -> bool {
      return DT.dominates(A, DT.getNode(NamesToBlocks.at(B)));
    };
    revng_check(DTNodeDominates(TheFiltDomTree, Sink, "initial_block"));
    revng_check(DTNodeDominates(TheFiltDomTree, Sink, "starter"));
    revng_check(DTNodeDominates(TheFiltDomTree, Sink, "false"));
    revng_check(DTNodeDominates(TheFiltDomTree, Sink, "end"));
    revng_check(DTNodeDominates(TheFiltDomTree, Sink, "_xit"));
    revng_check(not Dominates(TheFiltDomTree, "starter", "initial_block"));
  }

  // #### Edge Filtered Graphs ####

  { // Do-nothing filter
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames{
      "initial_block", "starter", "end", "false", "_xit",
    };
    std::vector<llvm::StringRef> Results;

    using EFG = EdgeFilteredGraph<DotGraph *, alwaysTrueEdge<DotGraph *>>;
    for (auto &It : llvm::depth_first(EFG(&Input)))
      Results.push_back(It->getName());
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I] == ExpectedResultsNames[I]);
  }
  { // Filter all, leaving no edge
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames{
      "initial_block",
    };
    std::vector<llvm::StringRef> Results;

    using EFG = EdgeFilteredGraph<DotGraph *, alwaysFalseEdge<DotGraph *>>;
    for (auto &It : llvm::depth_first(EFG(&Input)))
      Results.push_back(It->getName());
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I] == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards nodes with name without 'a's
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames = {
      "initial_block",
      "starter",
      "false",
    };
    std::vector<llvm::StringRef> Results;

    using EFG = EdgeFilteredGraph<DotGraph *, edgeHasLetterInName<'a'>>;
    for (auto &It : llvm::depth_first(EFG(&Input)))
      Results.push_back(It->getName());
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I] == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards nodes with name without 'e's
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames = {
      "initial_block",
      "starter",
      "end",
      "false",
    };
    std::vector<llvm::StringRef> Results;

    using EFG = EdgeFilteredGraph<DotGraph *, edgeHasLetterInName<'e'>>;
    for (auto &It : llvm::depth_first(EFG(&Input)))
      Results.push_back(It->getName());
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I] == ExpectedResultsNames[I]);
  }

  // #### Breadth first visits ####

  { // Do-nothing filter
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames{
      "initial_block", "starter", "end", "false", "_xit",
    };
    std::vector<llvm::StringRef> Results;

    using EFG = EdgeFilteredGraph<DotGraph *, alwaysTrueEdge<DotGraph *>>;
    for (auto &It : llvm::breadth_first(EFG(&Input)))
      Results.push_back(It->getName());
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I] == ExpectedResultsNames[I]);
  }
  { // Filter all, leaving no edge
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames{
      "initial_block",
    };
    std::vector<llvm::StringRef> Results;

    using EFG = EdgeFilteredGraph<DotGraph *, alwaysFalseEdge<DotGraph *>>;
    for (auto &It : llvm::breadth_first(EFG(&Input)))
      Results.push_back(It->getName());
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I] == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards nodes with name without 'a's
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames = {
      "initial_block",
      "starter",
      "false",
    };
    std::vector<llvm::StringRef> Results;

    using EFG = EdgeFilteredGraph<DotGraph *, edgeHasLetterInName<'a'>>;
    for (auto &It : llvm::breadth_first(EFG(&Input)))
      Results.push_back(It->getName());
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I] == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards nodes with name without 'e's
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames = {
      "initial_block",
      "starter",
      "end",
      "false",
    };
    std::vector<llvm::StringRef> Results;

    using EFG = EdgeFilteredGraph<DotGraph *, edgeHasLetterInName<'e'>>;
    for (auto &It : llvm::breadth_first(EFG(&Input)))
      Results.push_back(It->getName());
    revng_check(Results.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < Results.size(); ++I)
      revng_check(Results[I] == ExpectedResultsNames[I]);
  }

  // #### Inverse depth first visits ####
  { // Do-nothing filter
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames{
      "end",
      "starter",
      "initial_block",
      "false",
    };
    std::vector<llvm::StringRef> FirstResults, SecondResults;

    using EFG = EdgeFilteredGraph<DotNode *, alwaysTrueEdge<DotNode *>>;
    using InvEFG = llvm::Inverse<EFG>;

    DotNode *Exit = Input.getNodeByName("end");

    for (auto &It : llvm::depth_first(InvEFG(Exit)))
      FirstResults.push_back(It->getName());
    revng_check(FirstResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < FirstResults.size(); ++I)
      revng_check(FirstResults[I] == ExpectedResultsNames[I]);

    for (auto &It : llvm::inverse_depth_first(EFG(Exit)))
      SecondResults.push_back(It->getName());
    revng_check(SecondResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < SecondResults.size(); ++I)
      revng_check(SecondResults[I] == ExpectedResultsNames[I]);
  }
  { // Filter all, leaving no edge
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames{
      "end",
    };
    std::vector<llvm::StringRef> FirstResults, SecondResults;

    using EFG = EdgeFilteredGraph<DotNode *, alwaysFalseEdge<DotNode *>>;
    using InvEFG = llvm::Inverse<EFG>;

    DotNode *Exit = Input.getNodeByName("end");

    for (auto &It : llvm::depth_first(InvEFG(Exit)))
      FirstResults.push_back(It->getName().str());
    revng_check(FirstResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < FirstResults.size(); ++I)
      revng_check(FirstResults[I] == ExpectedResultsNames[I]);

    for (auto &It : llvm::inverse_depth_first(EFG(Exit)))
      SecondResults.push_back(It->getName().str());
    revng_check(SecondResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < SecondResults.size(); ++I)
      revng_check(SecondResults[I] == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards nodes with name without 'a's
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames{
      "end",
    };
    std::vector<llvm::StringRef> FirstResults, SecondResults;

    using EFG = EdgeFilteredGraph<DotNode *, edgeHasLetterInName<'a'>>;
    using InvEFG = llvm::Inverse<EFG>;

    DotNode *Exit = Input.getNodeByName("end");

    for (auto &It : llvm::depth_first(InvEFG(Exit)))
      FirstResults.push_back(It->getName());
    revng_check(FirstResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < FirstResults.size(); ++I)
      revng_check(FirstResults[I] == ExpectedResultsNames[I]);

    for (auto &It : llvm::inverse_depth_first(EFG(Exit)))
      SecondResults.push_back(It->getName());
    revng_check(SecondResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < SecondResults.size(); ++I)
      revng_check(SecondResults[I] == ExpectedResultsNames[I]);
  }
  { // Filter throwing away all the eges towards nodes BB with name without 'e's
    DotGraph Input;
    using namespace boost::unit_test::framework;
    revng_check(master_test_suite().argc == 2);
    std::string FileName = master_test_suite().argv[1];
    FileName += "001.dot";
    Input.parseDotFromFile(FileName, "initial_block");

    std::vector<llvm::StringRef> ExpectedResultsNames{
      "end",
      "starter",
      "initial_block",
      "false",
    };
    std::vector<llvm::StringRef> FirstResults, SecondResults;

    using EFG = EdgeFilteredGraph<DotNode *, edgeHasLetterInName<'e'>>;
    using InvEFG = llvm::Inverse<EFG>;

    DotNode *Exit = Input.getNodeByName("end");

    for (auto &It : llvm::depth_first(InvEFG(Exit)))
      FirstResults.push_back(It->getName());
    revng_check(FirstResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < FirstResults.size(); ++I)
      revng_check(FirstResults[I] == ExpectedResultsNames[I]);

    for (auto &It : llvm::inverse_depth_first(EFG(Exit)))
      SecondResults.push_back(It->getName());
    revng_check(SecondResults.size() == ExpectedResultsNames.size());
    for (size_t I = 0; I < SecondResults.size(); ++I)
      revng_check(SecondResults[I] == ExpectedResultsNames[I]);
  }

  // #### Dominator and PostDominator Trees ####
  // These are only on GraphTraits<DotNode *>, because they do not work on
  // GraphTraits<EdgeFilteredGraph<DotNode *, Filter>>
  // This is due to some quirks of how dominator and post dominator trees are
  // implemented that conflicts with how we expose predecessor and successor
  // edges in GraphTrait<DotNode *>.
  // Since nobody need this at the moment it has not been implemented, since it
  // would require heavy scrambling of the templates, which is not a priority
  // right now.
  // We might consider to fix it in the future if someone ends up needing it.
  DotGraph Input;
  using namespace boost::unit_test::framework;
  revng_check(master_test_suite().argc == 2);
  std::string FileName = master_test_suite().argv[1];
  FileName += "001.dot";
  Input.parseDotFromFile(FileName, "initial_block");

  const auto EFDominates =
    [&Input](auto &DT, const std::string &A, const std::string &B) -> bool {
    return DT.dominates(Input.getNodeByName(A), Input.getNodeByName(B));
  };
  {
    using DT = DominatorTreeBase<DotNode, false>;
    DT TheDomTree;
    TheDomTree.recalculate(Input);
    revng_check(TheDomTree.verify());
    TheDomTree.print(errs());
    revng_check(EFDominates(TheDomTree, "initial_block", "starter"));
    revng_check(EFDominates(TheDomTree, "starter", "end"));
    revng_check(EFDominates(TheDomTree, "starter", "false"));
    revng_check(EFDominates(TheDomTree, "false", "_xit"));
  }
  {
    using PostDT = DominatorTreeBase<DotNode, true>;
    PostDT PDT;
    PDT.recalculate(Input);
    revng_check(PDT.verify());
    PDT.print(errs());
    const auto *Sink = PDT.getRootNode();
    const auto DTNodeDominates = [&Input](auto &DT,
                                          const DomTreeNodeBase<DotNode> *N,
                                          const std::string &B) -> bool {
      return DT.dominates(N, DT.getNode(Input.getNodeByName(B)));
    };
    revng_check(DTNodeDominates(PDT, Sink, "initial_block"));
    revng_check(DTNodeDominates(PDT, Sink, "starter"));
    revng_check(DTNodeDominates(PDT, Sink, "false"));
    revng_check(DTNodeDominates(PDT, Sink, "end"));
    revng_check(DTNodeDominates(PDT, Sink, "_xit"));
    revng_check(EFDominates(PDT, "starter", "initial_block"));
  }
}

BOOST_AUTO_TEST_SUITE_END()
