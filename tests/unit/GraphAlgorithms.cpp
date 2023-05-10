/// \file GraphAlgorithms.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE GraphAlgorithms
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/ADT/SmallSet.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/GraphAlgorithms.h"

struct MyForwardNode {
  MyForwardNode(int Index) : Index(Index) {}
  int Index;
  int getIndex() { return Index; }
};

struct MyBidirectionalNode : MyForwardNode {
  MyBidirectionalNode(int Index) : MyForwardNode(Index) {}
};

template<typename NodeType>
struct LoopGraph {
  using Node = NodeType;
  GenericGraph<Node> Graph;
  Node *Entry;
  Node *LoopLatch;
  Node *Exit;
};

template<typename NodeType>
static LoopGraph<NodeType> createLGGraph() {
  LoopGraph<NodeType> LG;
  auto &Graph = LG.Graph;

  // Create nodes
  LG.Entry = Graph.addNode(1);
  LG.LoopLatch = Graph.addNode(2);
  LG.Exit = Graph.addNode(3);

  // Set entry node
  Graph.setEntryNode(LG.Entry);

  // Create edges
  LG.Entry->addSuccessor(LG.LoopLatch);
  LG.LoopLatch->addSuccessor(LG.Entry);
  LG.Entry->addSuccessor(LG.Exit);

  return LG;
}

template<typename NodeType>
struct NestedLoopGraph {
  using Node = NodeType;
  GenericGraph<Node> Graph;
  Node *Entry;
  Node *LoopHeader;
  Node *SecondLoopHeader;
  Node *LoopLatch;
  Node *SecondLoopLatch;
  Node *Exit;
};

template<typename NodeType>
static NestedLoopGraph<NodeType> createNLGGraph() {
  NestedLoopGraph<NodeType> NLG;
  auto &Graph = NLG.Graph;

  // Create nodes
  NLG.Entry = Graph.addNode(1);
  NLG.LoopHeader = Graph.addNode(2);
  NLG.SecondLoopHeader = Graph.addNode(3);
  NLG.LoopLatch = Graph.addNode(4);
  NLG.SecondLoopLatch = Graph.addNode(5);
  NLG.Exit = Graph.addNode(6);

  // Create edges
  NLG.Entry->addSuccessor(NLG.LoopHeader);
  NLG.LoopHeader->addSuccessor(NLG.SecondLoopHeader);
  NLG.SecondLoopHeader->addSuccessor(NLG.LoopLatch);
  NLG.LoopLatch->addSuccessor(NLG.SecondLoopLatch);
  NLG.LoopLatch->addSuccessor(NLG.SecondLoopHeader);
  NLG.SecondLoopLatch->addSuccessor(NLG.LoopHeader);
  NLG.SecondLoopLatch->addSuccessor(NLG.Exit);

  return NLG;
}

BOOST_AUTO_TEST_CASE(GetBackedgesTest) {
  // Create the graph
  using NodeType = ForwardNode<MyForwardNode>;
  auto LG = createLGGraph<NodeType>();
  using EdgeDescriptor = revng::detail::EdgeDescriptor<NodeType *>;
  using EdgeSet = llvm::SmallSetVector<EdgeDescriptor, 4>;
  using BlockSet = llvm::SmallSetVector<NodeType *, 4>;

  // Compute the backedges set
  EdgeSet Backedges = getBackedges(LG.Entry);

  // Check that the only backedge present
  revng_check(Backedges.size() == 1);
  EdgeDescriptor Backedge = *Backedges.begin();
  NodeType *Source = Backedge.first;
  NodeType *Target = Backedge.second;
  revng_check(Source == LG.LoopLatch);
  revng_check(Target == LG.Entry);

  // Check the reachability set described by the only backedge present
  BlockSet Reachables = nodesBetweenNew(Target, Source);
  revng_check(Reachables.size() == 2);
  revng_check(Reachables.contains(LG.Entry));
  revng_check(Reachables.contains(LG.LoopLatch));
  revng_check(LG.Entry != LG.LoopLatch);
  revng_check(Reachables[0] == LG.LoopLatch);
  revng_check(Reachables[1] == LG.Entry);
}

BOOST_AUTO_TEST_CASE(NestedLoopTest) {
  // Create the graph
  using NodeType = BidirectionalNode<MyBidirectionalNode>;
  auto NLG = createNLGGraph<NodeType>();
  using EdgeDescriptor = revng::detail::EdgeDescriptor<NodeType *>;
  using EdgeSet = llvm::SmallSetVector<EdgeDescriptor, 4>;
  using BlockSet = llvm::SmallSetVector<NodeType *, 4>;
  using BlockSetVect = llvm::SmallVector<BlockSet>;

  // Compute the backedges set
  EdgeSet Backedges = getBackedges(NLG.Entry);
  revng_check(Backedges.size() == 2);
  revng_check(Backedges[0].first == NLG.LoopLatch);
  revng_check(Backedges[0].second == NLG.SecondLoopHeader);
  revng_check(Backedges[1].first == NLG.SecondLoopLatch);
  revng_check(Backedges[1].second == NLG.LoopHeader);

  // Obtain the nodes reachable from the identified backedges
  BlockSetVect Loops;
  for (EdgeDescriptor Backedge : Backedges) {
    BlockSet LoopNodes = nodesBetweenNew(Backedge.second, Backedge.first);
    Loops.push_back(std::move(LoopNodes));
  }

  // We want to test that the nodes reachable from the extremities of a
  // backedge, compose the whole body of the loop. Two loops (one which is
  // completely nested inside the other) compose the test.
  revng_check(Loops.size() == 2);
  revng_check(Loops[0][0] == NLG.LoopLatch);
  revng_check(Loops[0][1] == NLG.SecondLoopHeader);
  revng_check(Loops[1][0] == NLG.SecondLoopLatch);
  revng_check(Loops[1][1] == NLG.LoopHeader);
  revng_check(Loops[1][2] == NLG.SecondLoopHeader);
  revng_check(Loops[1][3] == NLG.LoopLatch);
}
