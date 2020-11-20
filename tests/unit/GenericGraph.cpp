/// \file GenericGraph.cpp
/// \brief Test the GenericGraph

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#define BOOST_TEST_MODULE GenericGraph
bool init_unit_test();
#include "boost/test/unit_test.hpp"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/GenericDomTreeConstruction.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/FilteredGraphTraits.h"
#include "revng/ADT/GenericGraph.h"

using namespace llvm;

BOOST_AUTO_TEST_CASE(TestCompile) {
  // Test only it compiles
  if constexpr (false) {
    {
      struct MyForwardNode : public ForwardNode<MyForwardNode> {
        MyForwardNode(int) {}
        int m;
      };
      GenericGraph<MyForwardNode> Graph;
      auto *Node = Graph.addNode(3);
      Node->addSuccessor(Node);
      Node->addSuccessor(Node, {});
      MyForwardNode *Neighbor = *Node->successors().begin();
      Node->removeSuccessor(Node->successors().begin());
      Node->removeSuccessorEdge(Node->successor_edges().begin());
    }

    {
      struct MyBidirectionalNode
        : public BidirectionalNode<MyBidirectionalNode> {
        MyBidirectionalNode(int) {}
        int m;
      };
      GenericGraph<MyBidirectionalNode> Graph;
      auto *Node = Graph.addNode(3);
      Node->addSuccessor(Node);
      Node->addSuccessor(Node, {});
      Node->addPredecessor(Node);
      Node->addPredecessor(Node, {});
      MyBidirectionalNode *Neighbor = *Node->successors().begin();
      Neighbor = *Node->predecessors().begin();
      Node->removePredecessor(Node->predecessors().begin());
      Node->removePredecessorEdge(Node->predecessor_edges().begin());
    }

    struct EdgeLabel {
      int X;
    };

    {
      struct MyForwardNodeWithEdges
        : public ForwardNode<MyForwardNodeWithEdges, EdgeLabel> {
        MyForwardNodeWithEdges(int) {}
        int m;
      };

      auto [A, B] = Edge<MyForwardNodeWithEdges, EdgeLabel>{ nullptr };

      GenericGraph<MyForwardNodeWithEdges> Graph;
      auto *Node = Graph.addNode(3);
      Node->addSuccessor(Node);
      Node->addSuccessor(Node, { 99 });
      MyForwardNodeWithEdges *Neighbor = *Node->successors().begin();
    }

    {
      struct MyBidirectionalNodeWithEdges
        : public BidirectionalNode<MyBidirectionalNodeWithEdges, EdgeLabel> {
        MyBidirectionalNodeWithEdges(int) {}
        int m;
      };
      GenericGraph<MyBidirectionalNodeWithEdges> Graph;
      auto *Node = Graph.addNode(3);
      Node->addSuccessor(Node);
      Node->addSuccessor(Node, { 99 });
      Node->addPredecessor(Node);
      Node->addPredecessor(Node, { 99 });
      MyBidirectionalNodeWithEdges *Neighbor = *Node->successors().begin();
      Neighbor = *Node->predecessors().begin();
      Graph.removeNode(Graph.nodes().begin());

      using NGT = GraphTraits<MyBidirectionalNodeWithEdges *>;
      NGT::child_begin(Node);
      auto It = NGT::child_edge_begin(Node);

      Neighbor = NGT::edge_dest(*It);

      using INGT = GraphTraits<Inverse<MyBidirectionalNodeWithEdges *>>;
      INGT::child_begin(Node);

      using GGT = GraphTraits<GenericGraph<MyBidirectionalNodeWithEdges> *>;
      Graph.nodes();
    }
  }
}

struct TestEdgeLabel {
  unsigned Weight;
};

struct TestNode : public BidirectionalNode<TestNode, TestEdgeLabel> {
  TestNode(unsigned Rank) : Rank(Rank) {}
  unsigned Rank;
};

using TestGraph = GenericGraph<TestNode>;

static bool
shouldKeepNodePair(TestNode *const &Source, TestNode *const &Destination) {
  return Source->Rank + Destination->Rank <= 2;
}

static bool shouldKeepEdge(Edge<TestNode, TestEdgeLabel> &Edge) {
  return Edge.Weight > 5;
}

struct DiamondGraph {
  TestGraph Graph;
  TestNode *Root;
  TestNode *Then;
  TestNode *Else;
  TestNode *Final;
};

static DiamondGraph createGraph() {
  DiamondGraph DG;
  TestGraph &Graph = DG.Graph;

  // Create nodes
  DG.Root = Graph.addNode(0);
  DG.Then = Graph.addNode(1);
  DG.Else = Graph.addNode(1);
  DG.Final = Graph.addNode(2);

  // Set entry node
  Graph.setEntryNode(DG.Root);

  // Create edges
  DG.Root->addSuccessor(DG.Then, { 7 });
  DG.Root->addSuccessor(DG.Else, { 1 });

  DG.Then->addSuccessor(DG.Final, { 2 });
  DG.Else->addSuccessor(DG.Final, { 3 });

  return DG;
}

BOOST_AUTO_TEST_CASE(TestRPOT) {
  DiamondGraph DG = createGraph();
  ReversePostOrderTraversal<TestGraph *> RPOT(&DG.Graph);
  std::vector<TestNode *> Visited;
  for (TestNode *Node : RPOT)
    Visited.push_back(Node);
  revng_check(Visited.size() == 4);
}

BOOST_AUTO_TEST_CASE(TestDepthFirstVisit) {
  DiamondGraph DG = createGraph();
  std::vector<TestNode *> Visited;
  for (TestNode *Node : depth_first(&DG.Graph))
    Visited.push_back(Node);
  revng_check(Visited.size() == 4);

  Visited.clear();
  for (TestNode *Node : inverse_depth_first(DG.Final))
    Visited.push_back(Node);
  revng_check(Visited.size() == 4);
}

BOOST_AUTO_TEST_CASE(TestDominatorTree) {
  DiamondGraph DG = createGraph();

  DominatorTreeBase<TestNode, false> DT;
  DT.recalculate(DG.Graph);
  revng_check(DT.dominates(DT.getNode(DG.Root), DT.getNode(DG.Then)));
  revng_check(DT.dominates(DT.getNode(DG.Root), DT.getNode(DG.Else)));
  revng_check(DT.dominates(DT.getNode(DG.Root), DT.getNode(DG.Final)));
  revng_check(not DT.dominates(DT.getNode(DG.Then), DT.getNode(DG.Final)));

  DominatorTreeBase<TestNode, true> PDT;
  PDT.recalculate(DG.Graph);
  revng_check(PDT.dominates(PDT.getNode(DG.Final), PDT.getNode(DG.Then)));
  revng_check(PDT.dominates(PDT.getNode(DG.Final), PDT.getNode(DG.Else)));
  revng_check(PDT.dominates(PDT.getNode(DG.Final), PDT.getNode(DG.Root)));
  revng_check(not PDT.dominates(PDT.getNode(DG.Then), PDT.getNode(DG.Root)));
}

BOOST_AUTO_TEST_CASE(TestSCC) {
  DiamondGraph DG = createGraph();
  unsigned SCCCount = 0;
  for (const std::vector<TestNode *> &SCC :
       make_range(scc_begin(&DG.Graph), scc_end(&DG.Graph))) {
    revng_check(SCC.size() == 1);
    ++SCCCount;
  }
  revng_check(SCCCount == 4);
}

BOOST_AUTO_TEST_CASE(TestFilterGraphTraits) {
  DiamondGraph DG = createGraph();
  TestNode *Root = DG.Root;

  {
    using Pair = NodePairFilteredGraph<TestNode *, shouldKeepNodePair>;
    using FGT = GraphTraits<Pair>;
    using fdf_iterator = df_iterator<TestNode *,
                                     df_iterator_default_set<TestNode *>,
                                     false,
                                     FGT>;
    auto Begin = fdf_iterator::begin(Root);
    auto End = fdf_iterator::end(Root);
    revng_check(3 == std::distance(Begin, End));
  }

  {
    using EFGT = GraphTraits<EdgeFilteredGraph<TestNode *, shouldKeepEdge>>;
    using efdf_iterator = df_iterator<TestNode *,
                                      df_iterator_default_set<TestNode *>,
                                      false,
                                      EFGT>;
    auto Begin = efdf_iterator::begin(Root);
    auto End = efdf_iterator::end(Root);
    revng_check(2 == std::distance(Begin, End));
  }
}

BOOST_AUTO_TEST_CASE(TestWriteGraph) {
  DiamondGraph DG = createGraph();
  llvm::raw_null_ostream NullOutput;
  llvm::WriteGraph(NullOutput, &DG.Graph, "lol");
}
