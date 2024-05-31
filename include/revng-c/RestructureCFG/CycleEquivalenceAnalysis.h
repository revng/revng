#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <map>

#include "llvm/IR/CFG.h"
#include "llvm/Pass.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/Assert.h"

#include "revng-c/ADT/STLExtras.h"

#include "CycleEquivalenceClass.h"
#include "CycleEquivalenceResult.h"

enum class OriginalEdgeKind {
  Real,
  Fake,
  Invalid,
};

enum class SpanningTreeEdgeKind {
  TreeEdge,
  BackEdge,
  Invalid,
};

/// The `EdgeLabel` `struct` is used to represent the information needed to
/// identify each undirected edge in the `GenericGraph` materialized internally
/// in the analysis, and to map it eventually on the edge on the directed
/// original version of it. It has the following fields:
/// The `Kind` field is used to differentiate between the real edges (edges that
/// have a corresponde on the original directed graph, or additional edges like
/// capping backedges and the `exit` -> `entry` backedge).
/// The `EdgeType` is used to differentiate between `TreeEdge`s and `BackEdge`s,
/// after the computation of the spanning tree is performed.
/// The `ID`, an unique incremental identifier used to distinguish between the
/// edges.
/// The `SuccNum`, which represent the index which corresponds to the
/// represented edge in the successors of the `Source` node.
struct EdgeLabel {
private:
  static constexpr size_t SizeTMaxValue = std::numeric_limits<size_t>::max();

private:
  OriginalEdgeKind Kind = OriginalEdgeKind::Invalid;
  SpanningTreeEdgeKind Type = SpanningTreeEdgeKind::Invalid;

  // An incremental ID is necessary to disambiaguate when the `SuccNum` is not
  // enough, since on the undirected graph we may have a clash between such
  // types of edges: A,1 -> B and B,1 -> A, which on the undirected graph may
  // end up collapsed on the same identifier if we don't assign an additional
  // incremental identifier
  size_t ID = SizeTMaxValue;
  size_t SuccNum = SizeTMaxValue;

public:
  EdgeLabel(OriginalEdgeKind Kind,
            SpanningTreeEdgeKind Type,
            size_t ID,
            size_t SuccNum) :
    Kind(Kind), Type(Type), ID(ID), SuccNum(SuccNum) {}

  bool operator==(const EdgeLabel &) const = default;
  std::strong_ordering operator<=>(const EdgeLabel &) const = default;

  void setType(SpanningTreeEdgeKind Type) { this->Type = Type; }
  OriginalEdgeKind kind() const { return Kind; }
  SpanningTreeEdgeKind type() const { return Type; }

  size_t id() const {

    // We verify that the `ID` is not uninitialized
    revng_assert(ID != SizeTMaxValue);
    return ID;
  }

  size_t succNum() const {

    // We verify that the `SuccNum` is not uninitialized
    revng_assert(SuccNum != SizeTMaxValue);
    return SuccNum;
  }

  std::string edgeLabel() const {
    return std::to_string(SuccNum) + "," + std::to_string(ID);
  }

  std::string edgeStyle() const {
    revng_assert(Kind != OriginalEdgeKind::Invalid);
    revng_assert(Type != SpanningTreeEdgeKind::Invalid);

    if (Kind == OriginalEdgeKind::Real) {
      if (Type == SpanningTreeEdgeKind::TreeEdge) {

        // We want to print `TreeEdge`s as solid edges
        return "solid";
      } else if (Type == SpanningTreeEdgeKind::BackEdge) {

        // We want to print `Real` `BackeEdge`s as dashed
        return "dashed";
      }
    } else if (Kind == OriginalEdgeKind::Fake) {

      // `Fake` edges can be either capping `BackEdge`s or the edges connecting
      // the sink node (therefore they can be either a `TreeEdge` or a
      // `BackEdge`)
      return "dotted";
    }

    revng_abort("Kind or Type not expected");
  }
};

/// This class is used to compute the cycle equivalence classes over the graph
/// provided in input. Cycle equivalence is computed for all the edges in the
/// original graph.
/// Cycle equivalence is defined as follows: two edges in a graph are cycle
/// equivalent in a strongly connected component iff for all cycles C, C
/// contains either both edges or neither edge.
/// The algorithm to compute cycle equivalence is taken from "The Program
/// Structure Tree - Richard Johnson, David Pearson, Keshav Pingali - 1994"
/// https://dl.acm.org/doi/pdf/10.1145/178243.178258.
template<class GraphT, class GT = llvm::GraphTraits<GraphT>>
class CycleEquivalenceAnalysis {
private:
  static constexpr size_t SizeTMaxValue = std::numeric_limits<size_t>::max();

  // Incremental index for the edge `EdgeLabel`
  size_t IncrementalIndex = 0;
  size_t getIncrementalIndex() { return IncrementalIndex++; }

private:
  using NodeT = typename GT::NodeRef;

private:
  struct BlockBidirectionalNode {
    BlockBidirectionalNode(NodeT Block) : Block(Block) {}

    // This is used to store a pointer to the wrapped `NodeT`, except
    // for the optionally inserted `Sink` node, which does not correspond to any
    // node in the original graph
    NodeT Block;
    size_t DFSNum = SizeTMaxValue;

  public:
    NodeT getBlock() const {
      revng_assert(Block);
      return Block;
    }

    llvm::StringRef getName() const {

      // The only node with an associated `Block` field assigned to `nullptr`,
      // is the `Sink` node
      if (Block)
        return Block->getName();
      else
        return "sink";
    }

    void setDFSNum(size_t DFSNum) { this->DFSNum = DFSNum; }
    size_t getDFSNum() const { return DFSNum; }
  };

public:
  using BlockNode = MutableEdgeNode<BlockBidirectionalNode, EdgeLabel>;
  using BlockGraph = GenericGraph<BlockNode>;
  using BlockEdgeDescriptor = tuple_cat_t<
    revng::detail::EdgeDescriptor<BlockNode *>,
    make_tuple_t<size_t>>;
  using EdgesVectorType = llvm::SmallVector<
    std::tuple<BlockEdgeDescriptor, EdgeLabel *, bool>>;

private:
  // The `size_t` component in the pair represents the size of the open brackets
  // stack when the `BracketDescriptor` is created. The size is an integral and
  // necessary part to construct the unique identifier for the
  // `BracketDescriptor`.
  using BracketDescriptor = std::pair<BlockEdgeDescriptor, size_t>;

public:
  using CycleEquivalenceClassVector = llvm::SmallVector<
    CycleEquivalenceClass<NodeT>>;

  using CycleEquivalenceResult = CycleEquivalenceResult<NodeT>;

private:
  CycleEquivalenceClassVector CycleEquivalenceClasses;
  std::map<BracketDescriptor, size_t> BracketToClass;

  // This mapping is not strictly essential for the algorithm, but it is useful
  // to print the mapping wrt. to the `Bracket` which gave origin to a specific
  // equivalence class
  std::map<size_t, BracketDescriptor> ClassToBracketDescriptor;

  CycleEquivalenceResult EdgeToCycleEquivalenceClassIDMap;

public:
  CycleEquivalenceAnalysis() {}

  CycleEquivalenceResult getEdgeToCycleEquivalenceClassIDMap();

  void run(GraphT F);

private:
  bool isBackedge(BlockEdgeDescriptor &ED);

  void computeDFSAndSpanningTree(BlockGraph &Graph);

  void insertCappingBackedges(BlockGraph &Graph);

  void computeCappingValues(BlockNode *BB, size_t &HI0, size_t &HI2);

  EdgesVectorType getForwardEdges(BlockNode *Source);

  EdgesVectorType getBackwardEdges(BlockNode *Source);

  EdgesVectorType getTreeEdges(BlockNode *Source);

  void computeCycleEquivalence(const GraphT F, BlockGraph &Graph);

  BlockGraph initializeGenericGraph(const GraphT F);

  void insertEdge(BlockEdgeDescriptor E,
                  EdgeLabel *Label,
                  bool IsInverted,
                  BracketDescriptor BD);

  std::string print();
};

// Explicitly instantiate `DOTGraphTraits` for
// `CycleEquivalenceAnalysis<>::BlockGraph` instantiated for `llvm::Function *`
template<>
struct llvm::DOTGraphTraits<
  CycleEquivalenceAnalysis<llvm::Function *>::BlockGraph *>
  : public llvm::DefaultDOTGraphTraits {
  using llvm::DefaultDOTGraphTraits::DefaultDOTGraphTraits;
  using EdgeIterator = llvm::GraphTraits<CycleEquivalenceAnalysis<
    typename llvm::Function *>::BlockGraph *>::ChildIteratorType;
  using BlockNode = CycleEquivalenceAnalysis<llvm::Function *>::BlockNode;
  using BlockGraph = CycleEquivalenceAnalysis<llvm::Function *>::BlockGraph;

  std::string getNodeLabel(const BlockNode *N, const BlockGraph *G);

  std::string getEdgeAttributes(const BlockNode *,
                                const EdgeIterator EI,
                                const BlockGraph *G);
};

template<class GraphT>
CycleEquivalenceAnalysis<GraphT>::CycleEquivalenceResult
getEdgeToCycleEquivalenceClassIDMap(GraphT F);
