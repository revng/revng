//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE MFPExtraState
bool init_unit_test();
#include <set>
#include <string>
#include <vector>

#include "boost/test/unit_test.hpp"

#include "llvm/ADT/GraphTraits.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"

// Each node has a name and an ordered list of integer "operations" that the
// transfer function adds to a `std::set<int>` lattice. The `ExtraState`
// recording surface uses `(NodeName, OperationIndex)` as its key, so each
// individual operation can be observed before/after.
struct OperationNodeData {
  std::string Name;
  std::vector<int> Operations;
};

using OperationNode = ForwardNode<OperationNodeData>;
using OperationGraph = GenericGraph<OperationNode>;

using OperationKey = std::pair<std::string, size_t>;
using IntSet = std::set<int>;

struct OperationsMFI : public SetUnionLattice<IntSet> {
  using Label = OperationNode *;
  using GraphType = OperationGraph *;
  using LatticeElement = IntSet;
  using ExtraStateKey = OperationKey;
  using ExtraStateType = MFP::ExtraState<OperationKey, LatticeElement>;

  LatticeElement applyTransferFunction(Label Node,
                                       const LatticeElement &In,
                                       ExtraStateType &State) const {
    LatticeElement Current = In;
    for (size_t I = 0; I < Node->Operations.size(); ++I) {
      OperationKey K{ Node->Name, I };
      State.registerBefore(K, Current);
      Current.insert(Node->Operations[I]);
      State.registerAfter(K, Current);
    }
    return Current;
  }
};

static_assert(MFP::MonotoneFrameworkInstance<OperationsMFI>);

namespace {

struct DiamondGraph {
  // A diamond:
  //
  //         Entry:[1]
  //         /        \
  //   Left:[2]      Right:[3]
  //         \        /
  //           Tail:[4]
  //
  OperationGraph Graph;
  OperationNode *Entry = nullptr;
  OperationNode *Left = nullptr;
  OperationNode *Right = nullptr;
  OperationNode *Tail = nullptr;

  DiamondGraph() {
    Entry = Graph.addNode(OperationNodeData{ "Entry", { 1 } });
    Left = Graph.addNode(OperationNodeData{ "Left", { 2 } });
    Right = Graph.addNode(OperationNodeData{ "Right", { 3 } });
    Tail = Graph.addNode(OperationNodeData{ "Tail", { 4 } });
    Graph.setEntryNode(Entry);
    Entry->addSuccessor(Left);
    Entry->addSuccessor(Right);
    Left->addSuccessor(Tail);
    Right->addSuccessor(Tail);
  }
};

} // namespace

// The fixed point of a simple set-union analysis on a diamond graph is the
// union of every reachable predecessor's contributions. After convergence
// every node downstream of `Entry` must observe `1` on entry; `Tail` must see
// the contributions of both `Left` and `Right`.
BOOST_AUTO_TEST_CASE(DiamondMFP) {
  DiamondGraph G;

  OperationsMFI MFI;
  IntSet Bottom;
  IntSet ExtremalValue;
  std::vector<OperationNode *> ExtremalLabels{ G.Entry };

  MFP::MFPConfiguration<OperationsMFI> Configuration{
    .Instance = &MFI,
    .Flow = &G.Graph,
    .Bottom = &Bottom,
    .ExtremalValue = &ExtremalValue,
    .ExtremalLabels = &ExtremalLabels,
  };

  auto Result = MFP::getMaximalFixedPoint<OperationsMFI>(Configuration);

  BOOST_TEST(Result.at(G.Entry).InValue == IntSet{});
  BOOST_TEST(Result.at(G.Entry).OutValue == IntSet({ 1 }));

  BOOST_TEST(Result.at(G.Left).InValue == IntSet({ 1 }));
  BOOST_TEST(Result.at(G.Left).OutValue == IntSet({ 1, 2 }));

  BOOST_TEST(Result.at(G.Right).InValue == IntSet({ 1 }));
  BOOST_TEST(Result.at(G.Right).OutValue == IntSet({ 1, 3 }));

  // `Tail` joins both branches, so its incoming value must be `{1, 2, 3}` and
  // its outgoing value must additionally contain its own contribution `4`.
  BOOST_TEST(Result.at(G.Tail).InValue == IntSet({ 1, 2, 3 }));
  BOOST_TEST(Result.at(G.Tail).OutValue == IntSet({ 1, 2, 3, 4 }));
}

// Verify the analysis reaches the same fixed point on a graph with a back
// edge (and a multi-step node), exercising the worklist.
BOOST_AUTO_TEST_CASE(LoopMFP) {
  // A:[1] --> B:[2,3] --> C:[4]
  //              ^         /
  //              \________/
  OperationGraph Graph;
  auto *A = Graph.addNode(OperationNodeData{ "A", { 1 } });
  auto *B = Graph.addNode(OperationNodeData{ "B", { 2, 3 } });
  auto *C = Graph.addNode(OperationNodeData{ "C", { 4 } });
  Graph.setEntryNode(A);
  A->addSuccessor(B);
  B->addSuccessor(C);
  C->addSuccessor(B);

  OperationsMFI MFI;
  IntSet Bottom;
  IntSet ExtremalValue;
  std::vector<OperationNode *> ExtremalLabels{ A };

  MFP::MFPConfiguration<OperationsMFI> Configuration{
    .Instance = &MFI,
    .Flow = &Graph,
    .Bottom = &Bottom,
    .ExtremalValue = &ExtremalValue,
    .ExtremalLabels = &ExtremalLabels,
  };

  auto Result = MFP::getMaximalFixedPoint<OperationsMFI>(Configuration);

  // The loop forces B and C's incoming values to include {1, 2, 3, 4} once
  // the analysis converges.
  BOOST_TEST(Result.at(A).OutValue == IntSet({ 1 }));
  BOOST_TEST(Result.at(B).InValue == IntSet({ 1, 2, 3, 4 }));
  BOOST_TEST(Result.at(B).OutValue == IntSet({ 1, 2, 3, 4 }));
  BOOST_TEST(Result.at(C).InValue == IntSet({ 1, 2, 3, 4 }));
  BOOST_TEST(Result.at(C).OutValue == IntSet({ 1, 2, 3, 4 }));
}

// Same diamond as the first case, but the caller hands in an `ExtraState`
// pre-populated with a few interesting `(operation, position)` pairs and
// confirms the recorded values match the per-operation lattice values at the
// fixed point. The ExtraState observation is a check on top of the regular
// MFP result, not the focus of the test.
BOOST_AUTO_TEST_CASE(ExtraStateRecordingOnDiamond) {
  DiamondGraph G;

  using State = MFP::ExtraState<OperationKey, IntSet>;
  State S;

  // Mark a single point per node.
  OperationKey EntryOp0{ "Entry", 0 };
  OperationKey LeftOp0{ "Left", 0 };
  OperationKey TailOp0{ "Tail", 0 };
  S.registerAsInterestingBefore(EntryOp0);
  S.registerAsInterestingAfter(LeftOp0);
  S.registerAsInterestingBefore(TailOp0);

  OperationsMFI MFI;
  IntSet Bottom;
  IntSet ExtremalValue;
  std::vector<OperationNode *> ExtremalLabels{ G.Entry };

  MFP::MFPConfiguration<OperationsMFI> Configuration{
    .Instance = &MFI,
    .Flow = &G.Graph,
    .Bottom = &Bottom,
    .ExtremalValue = &ExtremalValue,
    .ExtremalLabels = &ExtremalLabels,
    .ExtraState = &S
  };

  auto Result = MFP::getMaximalFixedPoint<OperationsMFI>(Configuration);

  // Sanity: the fixed-point lattice values must match the diamond test above.
  BOOST_TEST(Result.at(G.Tail).OutValue == IntSet({ 1, 2, 3, 4 }));

  // The recorded values agree with the labels' InValue / partial OutValue.
  BOOST_TEST(S.getBefore(EntryOp0) == IntSet{});
  BOOST_TEST(S.getAfter(LeftOp0) == IntSet({ 1, 2 }));
  BOOST_TEST(S.getBefore(TailOp0) == IntSet({ 1, 2, 3 }));
}
