/// \file GraphAlgorithms.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#define BOOST_TEST_MODULE GraphAlgorithms
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/ADT/SmallSet.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/GraphAlgorithms.h"

struct Block {
  Block(int Index) : Index(Index) {}
  int Index;
  int getIndex() { return Index; }
};

struct MyBidirectionalNode : Block {
  MyBidirectionalNode(int Index) : Block(Index) {}
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

template<typename NodeType>
struct NonCanonicalLoopGraph {
  using Node = NodeType;
  GenericGraph<Node> Graph;
  Node *LoopHeader;
  Node *SecondLoopHeader;
  Node *LoopLatch;
  Node *LoopSuccessor;
  Node *SecondLoopSuccessor;
  Node *SecondLoopLatch;
};

template<typename NodeType>
static NonCanonicalLoopGraph<NodeType> createNCLGGraph() {
  NonCanonicalLoopGraph<NodeType> NCLG;
  auto &Graph = NCLG.Graph;

  // Create nodes
  NCLG.LoopHeader = Graph.addNode(1);
  NCLG.SecondLoopHeader = Graph.addNode(2);
  NCLG.LoopLatch = Graph.addNode(3);
  NCLG.LoopSuccessor = Graph.addNode(4);
  NCLG.SecondLoopSuccessor = Graph.addNode(5);
  NCLG.SecondLoopLatch = Graph.addNode(6);

  // Create edges
  NCLG.LoopHeader->addSuccessor(NCLG.SecondLoopHeader);
  NCLG.SecondLoopHeader->addSuccessor(NCLG.LoopLatch);
  NCLG.SecondLoopHeader->addSuccessor(NCLG.LoopSuccessor);
  NCLG.LoopLatch->addSuccessor(NCLG.LoopSuccessor);
  NCLG.LoopSuccessor->addSuccessor(NCLG.SecondLoopSuccessor);
  NCLG.LoopSuccessor->addSuccessor(NCLG.SecondLoopLatch);
  NCLG.SecondLoopSuccessor->addSuccessor(NCLG.SecondLoopLatch);
  NCLG.LoopLatch->addSuccessor(NCLG.LoopHeader);
  NCLG.SecondLoopLatch->addSuccessor(NCLG.SecondLoopHeader);
  return NCLG;
}

template<typename NodeType>
struct NonCanonicalAlternateLoopGraph {
  using Node = NodeType;
  GenericGraph<Node> Graph;
  Node *LoopHeader;
  Node *SecondLoopHeader;
  Node *LoopLatch;
  Node *LoopSuccessor;
  Node *SecondLoopSuccessor;
  Node *SecondLoopLatch;
};

template<typename NodeType>
static NonCanonicalAlternateLoopGraph<NodeType> createNCALGGraph() {
  NonCanonicalAlternateLoopGraph<NodeType> NCALG;
  auto &Graph = NCALG.Graph;

  // Create nodes
  NCALG.LoopHeader = Graph.addNode(1);
  NCALG.SecondLoopHeader = Graph.addNode(2);
  NCALG.LoopLatch = Graph.addNode(3);
  NCALG.SecondLoopLatch = Graph.addNode(4);

  // Create edges
  NCALG.LoopHeader->addSuccessor(NCALG.SecondLoopHeader);
  NCALG.SecondLoopHeader->addSuccessor(NCALG.LoopLatch);
  NCALG.LoopLatch->addSuccessor(NCALG.SecondLoopLatch);
  NCALG.LoopLatch->addSuccessor(NCALG.LoopHeader);
  NCALG.SecondLoopLatch->addSuccessor(NCALG.SecondLoopHeader);
  return NCALG;
}

template<typename NodeType>
struct ReverseGraph {
  using Node = NodeType;
  GenericGraph<Node> Graph;
  Node *InitialBlock;
  Node *SmallerBlock;
  Node *EndBlock;
};

template<typename NodeType>
static ReverseGraph<NodeType> createRGraph() {
  ReverseGraph<NodeType> RG;
  auto &Graph = RG.Graph;

  // Create nodes
  RG.InitialBlock = Graph.addNode(1);
  RG.SmallerBlock = Graph.addNode(2);
  RG.EndBlock = Graph.addNode(3);

  // Create edges
  RG.InitialBlock->addSuccessor(RG.SmallerBlock);
  RG.InitialBlock->addSuccessor(RG.EndBlock);
  RG.SmallerBlock->addSuccessor(RG.EndBlock);
  return RG;
}

BOOST_AUTO_TEST_CASE(GetBackedgesTest) {
  // Create the graph
  using NodeType = BidirectionalNode<MyBidirectionalNode>;
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
  BlockSet Reachables = nodesBetween(Target, Source);
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
    BlockSet LoopNodes = nodesBetween(Backedge.second, Backedge.first);
    Loops.push_back(std::move(LoopNodes));
  }

  // We want to test that the nodes reachable from the extremities of a
  // backedge, compose the whole body of the loop. Two loops (one which is
  // completely nested inside the other) compose the test.
  revng_check(Loops.size() == 2);
  revng_check(Loops[0][0] == NLG.LoopLatch);
  revng_check(Loops[0][1] == NLG.SecondLoopHeader);
  revng_check(Loops[1][0] == NLG.LoopLatch);
  revng_check(Loops[1][1] == NLG.SecondLoopHeader);
  revng_check(Loops[1][2] == NLG.SecondLoopLatch);
  revng_check(Loops[1][3] == NLG.LoopHeader);
}

BOOST_AUTO_TEST_CASE(NonCanonicalLoopTest) {
  // Create the graph
  using NodeType = BidirectionalNode<MyBidirectionalNode>;
  auto NCLG = createNCLGGraph<NodeType>();
  using EdgeDescriptor = revng::detail::EdgeDescriptor<NodeType *>;
  using EdgeSet = llvm::SmallSetVector<EdgeDescriptor, 4>;
  using BlockSet = llvm::SmallSetVector<NodeType *, 4>;
  using BlockSetVect = llvm::SmallVector<BlockSet>;

  // Compute the backedges set
  EdgeSet Backedges = getBackedges(NCLG.LoopHeader);
  revng_check(Backedges.size() == 2);
  revng_check(Backedges[0].first == NCLG.LoopLatch);
  revng_check(Backedges[0].second == NCLG.LoopHeader);
  revng_check(Backedges[1].first == NCLG.SecondLoopLatch);
  revng_check(Backedges[1].second == NCLG.SecondLoopHeader);

  // Obtain the nodes reachable from the identified backedges
  BlockSetVect Loops;
  for (EdgeDescriptor Backedge : Backedges) {
    BlockSet LoopNodes = nodesBetween(Backedge.second, Backedge.first);
    Loops.push_back(std::move(LoopNodes));
  }

  // We want to test that the nodes reachable from the extremities of a
  // backedge, compose the whole body of the loop. In particular, we want to
  // ensure that both `LoopSuccessor` and `SecondLoopSuccessor` are correctly
  // included in the first loop.
  revng_check(Loops.size() == 2);

  revng_check(Loops[0][0] == NCLG.SecondLoopHeader);
  revng_check(Loops[0][1] == NCLG.SecondLoopLatch);
  revng_check(Loops[0][2] == NCLG.LoopSuccessor);
  revng_check(Loops[0][3] == NCLG.SecondLoopSuccessor);
  revng_check(Loops[0][4] == NCLG.LoopLatch);
  revng_check(Loops[0][5] == NCLG.LoopHeader);

  revng_check(Loops[1][0] == NCLG.LoopSuccessor);
  revng_check(Loops[1][1] == NCLG.LoopLatch);
  revng_check(Loops[1][2] == NCLG.SecondLoopSuccessor);
  revng_check(Loops[1][3] == NCLG.SecondLoopLatch);
  revng_check(Loops[1][4] == NCLG.SecondLoopHeader);
}

BOOST_AUTO_TEST_CASE(NonCanonicalAlternateLoopTest) {
  // Create the graph
  using NodeType = BidirectionalNode<MyBidirectionalNode>;
  auto NCALG = createNCALGGraph<NodeType>();
  using EdgeDescriptor = revng::detail::EdgeDescriptor<NodeType *>;
  using EdgeSet = llvm::SmallSetVector<EdgeDescriptor, 4>;
  using BlockSet = llvm::SmallSetVector<NodeType *, 4>;
  using BlockSetVect = llvm::SmallVector<BlockSet>;

  // Compute the backedges set
  EdgeSet Backedges = getBackedges(NCALG.LoopHeader);
  revng_check(Backedges.size() == 2);
  revng_check(Backedges[0].first == NCALG.LoopLatch);
  revng_check(Backedges[0].second == NCALG.LoopHeader);
  revng_check(Backedges[1].first == NCALG.SecondLoopLatch);
  revng_check(Backedges[1].second == NCALG.SecondLoopHeader);

  // Obtain the nodes reachable from the identified backedges
  BlockSetVect Loops;
  for (EdgeDescriptor Backedge : Backedges) {
    BlockSet LoopNodes = nodesBetween(Backedge.second, Backedge.first);
    Loops.push_back(std::move(LoopNodes));
  }

  // We want to test that the nodes reachable from the extremities of a
  // backedge, compose the whole body of the loop. The two loops are partially
  // overlapping, but we don't want to trepass the head of the loop when
  // invoking the `nodesBetween`.
  revng_check(Loops.size() == 2);

  revng_check(Loops[0][0] == NCALG.SecondLoopHeader);
  revng_check(Loops[0][1] == NCALG.LoopLatch);
  revng_check(Loops[0][2] == NCALG.LoopHeader);

  revng_check(Loops[1][0] == NCALG.LoopLatch);
  revng_check(Loops[1][1] == NCALG.SecondLoopLatch);
  revng_check(Loops[1][2] == NCALG.SecondLoopHeader);
}

BOOST_AUTO_TEST_CASE(ReverseTest) {
  // Create the graph
  using NodeType = BidirectionalNode<MyBidirectionalNode>;
  auto RG = createRGraph<NodeType>();
  using EdgeDescriptor = revng::detail::EdgeDescriptor<NodeType *>;
  using EdgeSet = llvm::SmallSetVector<EdgeDescriptor, 4>;
  using BlockSet = llvm::SmallSetVector<NodeType *, 4>;

  // Compute the backedges set
  EdgeSet Backedges = getBackedges(RG.InitialBlock);
  revng_check(Backedges.size() == 0);

  // Test the reverse `nodesBetween`
  BlockSet ReachableSet = nodesBetweenReverse(RG.SmallerBlock, RG.InitialBlock);
  revng_check(ReachableSet.size() == 2);
  revng_check(ReachableSet[0] == RG.InitialBlock);
  revng_check(ReachableSet[1] == RG.SmallerBlock);
}

BOOST_AUTO_TEST_CASE(NullTargetTest) {
  // Create the graph
  using NodeType = BidirectionalNode<MyBidirectionalNode>;
  auto RG = createRGraph<NodeType>();
  using EdgeDescriptor = revng::detail::EdgeDescriptor<NodeType *>;
  using EdgeSet = llvm::SmallSetVector<EdgeDescriptor, 4>;
  using BlockSet = llvm::SmallSetVector<NodeType *, 4>;

  // Compute the backedges set
  EdgeSet Backedges = getBackedges(RG.InitialBlock);
  revng_check(Backedges.size() == 0);

  // Test the reverse `nodesBetween`
  BlockSet ReachableSet = findReachableNodes(RG.InitialBlock);
  revng_check(ReachableSet.size() == 3);
  revng_check(ReachableSet[0] == RG.InitialBlock);
  revng_check(ReachableSet[1] == RG.SmallerBlock);
  revng_check(ReachableSet[2] == RG.EndBlock);
}
