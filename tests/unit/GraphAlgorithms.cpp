/// \file GraphAlgorithms.cpp
/// \brief Test the GraphAlgorithms utils

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE GraphAlgorithms
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/ADT/SmallSet.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/GraphAlgorithms.h"

using namespace llvm;

struct MyForwardNode {
  MyForwardNode(int Index) : Index(Index) {}
  int Index;
  int getIndex() { return Index; }
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
struct OverLappingLoopGraph {
  using Node = NodeType;
  GenericGraph<Node> Graph;
  Node *Entry;
  Node *SecondEntry;
  Node *Latch;
  Node *SecondLatch;
  Node *Exit;
};

template<typename NodeType>
static OverLappingLoopGraph<NodeType> createOLGGraph() {
  OverLappingLoopGraph<NodeType> OLG;
  auto &Graph = OLG.Graph;

  // Create nodes
  OLG.Entry = Graph.addNode(1);
  OLG.SecondEntry = Graph.addNode(2);
  OLG.Latch = Graph.addNode(3);
  OLG.SecondLatch = Graph.addNode(4);
  OLG.Exit = Graph.addNode(5);

  // Create edges
  OLG.Entry->addSuccessor(OLG.SecondEntry);
  OLG.SecondEntry->addSuccessor(OLG.Latch);
  OLG.Latch->addSuccessor(OLG.SecondLatch);
  OLG.Latch->addSuccessor(OLG.Entry);
  OLG.SecondLatch->addSuccessor(OLG.SecondEntry);
  OLG.SecondLatch->addSuccessor(OLG.Exit);

  return OLG;
}

template<typename NodeType>
struct NestedLoopGraph {
  using Node = NodeType;
  GenericGraph<Node> Graph;
  Node *Entry;
  Node *SecondEntry;
  Node *Latch;
  Node *SecondLatch;
  Node *Exit;
};

template<typename NodeType>
static NestedLoopGraph<NodeType> createNLGGraph() {
  NestedLoopGraph<NodeType> NLG;
  auto &Graph = NLG.Graph;

  // Create nodes
  NLG.Entry = Graph.addNode(1);
  NLG.SecondEntry = Graph.addNode(2);
  NLG.Latch = Graph.addNode(3);
  NLG.SecondLatch = Graph.addNode(4);
  NLG.Exit = Graph.addNode(5);

  // Create edges
  NLG.Entry->addSuccessor(NLG.SecondEntry);
  NLG.SecondEntry->addSuccessor(NLG.Latch);
  NLG.Latch->addSuccessor(NLG.SecondLatch);
  NLG.Latch->addSuccessor(NLG.SecondEntry);
  NLG.SecondLatch->addSuccessor(NLG.Entry);
  NLG.SecondLatch->addSuccessor(NLG.Exit);

  return NLG;
}

template<typename NodeType>
static NestedLoopGraph<NodeType> createINLGGraph() {
  NestedLoopGraph<NodeType> INLG = createNLGGraph<NodeType>();
  auto &Graph = INLG.Graph;

  // Create forward inling edge.
  INLG.Entry->addSuccessor(INLG.Latch);

  return INLG;
}

template<class NodeType>
void printEdge(revng::detail::EdgeDescriptor<NodeType *> &Backedge) {
  llvm::dbgs() << "Backedge: ";
  llvm::dbgs() << Backedge.first->getIndex();
  llvm::dbgs() << " -> ";
  llvm::dbgs() << Backedge.second->getIndex();
  llvm::dbgs() << "\n";
}

template<class NodeType>
void printRegion(llvm::SmallPtrSet<NodeType *, 4> &Region) {
  for (auto *Block : Region) {
    llvm::dbgs() << Block->getIndex() << "\n";
  }
}

template<class NodeType>
void printRegions(llvm::SmallVector<llvm::SmallPtrSet<NodeType *, 4>, 4> &Rs) {
  using BlockSet = llvm::SmallPtrSet<NodeType *, 4>;
  size_t RegionIndex = 0;
  for (BlockSet &Region : Rs) {
    llvm::dbgs() << "Region idx: " << RegionIndex << " composed by nodes: \n";
    printRegion(Region);
    RegionIndex++;
  }
}

BOOST_AUTO_TEST_CASE(GetBackedgesTest) {
  // Create the graph.
  using NodeType = ForwardNode<MyForwardNode>;
  auto LG = createLGGraph<NodeType>();
  using EdgeDescriptor = revng::detail::EdgeDescriptor<NodeType *>;
  using EdgeSet = llvm::SmallSet<EdgeDescriptor, 4>;
  using BlockSet = llvm::SmallPtrSet<NodeType *, 4>;

  // Compute the backedges set.
  EdgeSet Backedges = getBackedges(LG.Entry);

  // Check that the only backedge present.
  revng_check(Backedges.size() == 1);
  EdgeDescriptor Backedge = *Backedges.begin();
  NodeType *Source = Backedge.first;
  NodeType *Target = Backedge.second;
  revng_check(Source == LG.LoopLatch);
  revng_check(Target == LG.Entry);

  // Check the reachability set described by the only backedge present.
  BlockSet Reachables = nodesBetween(Target, Source);
  revng_check(Reachables.size() == 2);
  revng_check(Reachables.contains(LG.Entry));
  revng_check(Reachables.contains(LG.LoopLatch));
  revng_check(LG.Entry != LG.LoopLatch);
}
