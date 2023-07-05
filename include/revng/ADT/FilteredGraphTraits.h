#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

// LLVM Includes
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/iterator_range.h"

//
// Filtered views on graphs, with predicates on node pairs representing edges
//

template<typename GraphType, typename PredicateType>
struct NodePairFilteredGraphImpl {
protected:
  using BaseGraphTraits = llvm::GraphTraits<GraphType>;

  const GraphType &Graph;

public:
  inline explicit NodePairFilteredGraphImpl(const GraphType &G) : Graph(G) {}
  inline explicit operator const GraphType &() const { return Graph; }

  static auto filter() { return PredicateType::value; }

  // NodeRef are the same as the BaseGraphTraits, because the graph has the same
  // kind of nodes

  // typedef NodeRef           - Type of Node token in the graph, which should
  //                             be cheap to copy.
  using NodeRef = typename BaseGraphTraits::NodeRef;

  // typedef ChildEdgeIteratorType - Type used to iterate over children edges in
  //                             graph, dereference to a EdgeRef.

protected:
  using PredT = typename PredicateType::value_type;
  static_assert(std::is_invocable_r_v<bool, PredT, NodeRef, NodeRef>);

public:
  // static NodeRef getEntryNode(const GraphType &)
  //    Return the entry node of the graph
  static NodeRef getEntryNode(const NodePairFilteredGraphImpl &G) {
    return BaseGraphTraits::getEntryNode(G.Graph);
  }

  static const NodeRef &getEntryNode(const NodeRef &N) { return N; }

  // static ChildIteratorType child_begin(NodeRef)
  // static ChildIteratorType child_end  (NodeRef)
  //    Return iterators that point to the beginning and ending of the child
  //    node list for the specified node.
  static auto child_begin(NodeRef N) {
    auto Children = llvm::make_range(BaseGraphTraits::child_begin(N),
                                     BaseGraphTraits::child_end(N));
    const auto P = std::bind(PredicateType::value, N, std::placeholders::_1);
    return llvm::make_filter_range(Children, std::move(P)).begin();
  }
  static auto child_end(NodeRef N) {
    auto Children = llvm::make_range(BaseGraphTraits::child_begin(N),
                                     BaseGraphTraits::child_end(N));
    const auto P = std::bind(PredicateType::value, N, std::placeholders::_1);
    return llvm::make_filter_range(Children, std::move(P)).end();
  }

  static_assert(std::is_same_v<decltype(child_begin(std::declval<NodeRef>())),
                               decltype(child_end(std::declval<NodeRef>()))>);

  using ChildIteratorType = decltype(child_begin(std::declval<NodeRef>()));
};

template<typename GraphType, typename PredicateType>
struct llvm::Inverse<NodePairFilteredGraphImpl<GraphType, PredicateType>> {
protected:
  using NPFGT = NodePairFilteredGraphImpl<GraphType, PredicateType>;
  using BaseGraphTraits = llvm::GraphTraits<GraphType>;
  using InvGraphTraits = llvm::GraphTraits<llvm::Inverse<GraphType>>;

  const GraphType &Graph;

public:
  inline explicit Inverse(const GraphType &G) : Graph(G) {}
  inline explicit Inverse(const NPFGT &FG) : Graph(FG) {}

  // typedef NodeRef           - Type of Node token in the graph, which should
  //                             be cheap to copy.
  using NodeRef = typename BaseGraphTraits::NodeRef;

  // typedef ChildEdgeIteratorType - Type used to iterate over children edges in
  //                             graph, dereference to a EdgeRef.

protected:
  using PredT = typename PredicateType::value_type;
  static_assert(std::is_invocable_r_v<bool, PredT, NodeRef, NodeRef>);

public:
  // static NodeRef getEntryNode(const GraphType &)
  //    Return the entry node of the graph
  static NodeRef getEntryNode(const Inverse &G) {
    return InvGraphTraits::getEntryNode(G.Graph);
  }

  // static ChildIteratorType child_begin(NodeRef)
  // static ChildIteratorType child_end  (NodeRef)
  //    Return iterators that point to the beginning and ending of the child
  //    node list for the specified node.
  static auto child_begin(NodeRef N) {
    auto Children = llvm::make_range(InvGraphTraits::child_begin(N),
                                     InvGraphTraits::child_end(N));
    const auto P = std::bind(PredicateType::value, std::placeholders::_1, N);
    return llvm::make_filter_range(Children, std::move(P)).begin();
  }
  static auto child_end(NodeRef N) {
    auto Children = llvm::make_range(InvGraphTraits::child_begin(N),
                                     InvGraphTraits::child_end(N));
    const auto P = std::bind(PredicateType::value, std::placeholders::_1, N);
    return llvm::make_filter_range(Children, std::move(P)).end();
  }

  static_assert(std::is_same_v<decltype(child_begin(std::declval<NodeRef>())),
                               decltype(child_end(std::declval<NodeRef>()))>);

  using ChildIteratorType = decltype(child_begin(std::declval<NodeRef>()));
};

// Provides a filtered view over llvm::GraphTraits<GraphType>.
// PredicateType must be a std::integral_constant wrapping a callable takes two
// arguments of type GraphType::NodeRef (the source and the target of an edge),
// and computes a boolean predicate.
// This view has all the nodes of the underlying graph, but only the edges for
// which the predicate is true.
template<typename GraphType, typename Pred>
struct llvm::GraphTraits<NodePairFilteredGraphImpl<GraphType, Pred>>
  : public NodePairFilteredGraphImpl<GraphType, Pred> {};

template<typename GT, typename Pred>
struct llvm::GraphTraits<llvm::Inverse<NodePairFilteredGraphImpl<GT, Pred>>>
  : public llvm::Inverse<NodePairFilteredGraphImpl<GT, Pred>> {};

template<typename NodeT>
using NodePairFilter =
  bool (*)(const typename llvm::GraphTraits<NodeT>::NodeRef &,
           const typename llvm::GraphTraits<NodeT>::NodeRef &);

template<class T, T v>
using ic = std::integral_constant<T, v>;

template<typename T>
using iterator_value = std::remove_reference_t<decltype(*std::declval<T>())>;

template<typename NodeT, NodePairFilter<NodeT> F>
using NodePairFilteredGraph = NodePairFilteredGraphImpl<NodeT,
                                                        ic<decltype(F), F>>;

//
// Filtered views on graphs, with predicates on edges
//

template<typename GraphType, typename PredicateType>
struct EdgeFilteredGraphImpl {
protected:
  using BaseGraphTraits = llvm::GraphTraits<GraphType>;

  const GraphType &Graph;

public:
  inline explicit EdgeFilteredGraphImpl(const GraphType &G) : Graph(G) {}
  inline explicit operator const GraphType &() const { return Graph; }

  static auto filter() { return PredicateType::value; }

  // NodeRef and EdgeRef are the same as the BaseGraphTraits, because the graph
  // has the same kind of nodes and edges

  // typedef NodeRef           - Type of Node token in the graph, which should
  //                             be cheap to copy.
  using NodeRef = typename BaseGraphTraits::NodeRef;

  // typedef EdgeRef           - Type of Edge token in the graph, which should
  //                             be cheap to copy.
  using EdgeRef = typename BaseGraphTraits::EdgeRef;

  // typedef ChildEdgeIteratorType - Type used to iterate over children edges in
  //                                 graph, dereference to a EdgeRef.

protected:
  using PredT = typename PredicateType::value_type;
  static_assert(std::is_invocable_r_v<bool, PredT, EdgeRef>);

  using BaseChildEdgeIt = typename BaseGraphTraits::ChildEdgeIteratorType;

public:
  using ChildEdgeIteratorType = llvm::filter_iterator<BaseChildEdgeIt, PredT>;
  static_assert(std::is_same_v<std::remove_reference_t<EdgeRef>,
                               iterator_value<ChildEdgeIteratorType>>);

  // static ChildEdgeIteratorType child_edge_begin(NodeRef)
  // static ChildEdgeIteratorType child_edge_end(NodeRef)
  //     Return iterators that point to the beginning and ending of the
  //     edge list for the given callgraph node.
  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    auto Rng = llvm::make_range(BaseGraphTraits::child_edge_begin(N),
                                BaseGraphTraits::child_edge_end(N));
    return llvm::make_filter_range(Rng, PredicateType::value).begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) {
    auto Rng = llvm::make_range(BaseGraphTraits::child_edge_begin(N),
                                BaseGraphTraits::child_edge_end(N));
    return llvm::make_filter_range(Rng, PredicateType::value).end();
  }

  // static NodeRef edge_dest(EdgeRef)
  //     Return the destination node of an edge.
  static NodeRef edge_dest(EdgeRef E) { return BaseGraphTraits::edge_dest(E); }

  // typedef ChildIteratorType - Type used to iterate over children in graph,
  //                             dereference to a NodeRef.

  // static NodeRef getEntryNode(const GraphType &)
  //    Return the entry node of the graph
  static NodeRef getEntryNode(const EdgeFilteredGraphImpl &G) {
    return BaseGraphTraits::getEntryNode(G.Graph);
  }

  static const NodeRef &getEntryNode(const NodeRef &N) { return N; }

  using ChildIteratorType = llvm::mapped_iterator<ChildEdgeIteratorType,
                                                  NodeRef (*)(EdgeRef)>;
  static_assert(std::is_same_v<NodeRef, iterator_value<ChildIteratorType>>);
  // static ChildIteratorType child_begin(NodeRef)
  // static ChildIteratorType child_end  (NodeRef)
  //    Return iterators that point to the beginning and ending of the child
  //    node list for the specified node.
  static ChildIteratorType child_begin(NodeRef N) {
    return llvm::map_iterator(child_edge_begin(N), edge_dest);
  }
  static ChildIteratorType child_end(NodeRef N) {
    return llvm::map_iterator(child_edge_end(N), edge_dest);
  }

  // typedef  ...iterator nodes_iterator; - dereference to a NodeRef
  // static nodes_iterator nodes_begin(GraphType *G)
  // static nodes_iterator nodes_end  (GraphType *G)
  //    nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  // using nodes_iterator = typename BaseGraphTraits::nodes_iterator;
  // static nodes_iterator nodes_begin(EdgeFilteredGraphImpl *G) {
  //   return BaseGraphTraits::nodes_begin(G->Graph);
  // }
  // static nodes_iterator nodes_end  (EdgeFilteredGraphImpl *G) {
  //   return BaseGraphTraits::nodes_end(G->Graph);
  // }

  // // static unsigned       size       (GraphType *G)
  // //    Return total number of nodes in the graph
  // static unsigned size(EdgeFilteredGraphImpl *G) {
  //   return BaseGraphTraits::size(G->Graph);
  // }
};

template<typename GraphType, typename PredicateType>
struct llvm::Inverse<EdgeFilteredGraphImpl<GraphType, PredicateType>> {
protected:
  using EFGT = EdgeFilteredGraphImpl<GraphType, PredicateType>;
  using BaseGraphTraits = llvm::GraphTraits<GraphType>;
  using InvGraphTraits = llvm::GraphTraits<llvm::Inverse<GraphType>>;

  const GraphType &Graph;

public:
  inline explicit Inverse(const GraphType &G) : Graph(G) {}
  inline explicit Inverse(const EFGT &FG) : Graph(FG) {}

  // NodeRef and EdgeRef are the same as the BaseGraphTraits, because the graph
  // has the same kind of nodes and edges

  // typedef NodeRef           - Type of Node token in the graph, which should
  //                             be cheap to copy.
  using NodeRef = typename BaseGraphTraits::NodeRef;

  // typedef EdgeRef           - Type of Edge token in the graph, which should
  //                             be cheap to copy.
  using EdgeRef = typename BaseGraphTraits::EdgeRef;

  // typedef ChildEdgeIteratorType - Type used to iterate over children edges in
  //                             graph, dereference to a EdgeRef.

protected:
  using PredT = typename PredicateType::value_type;
  static_assert(std::is_invocable_r_v<bool, PredT, EdgeRef>);

  using BaseChildEdgeIt = typename BaseGraphTraits::ChildEdgeIteratorType;
  using InvChildEdgeIt = typename InvGraphTraits::ChildEdgeIteratorType;

public:
  using ChildEdgeIteratorType = llvm::filter_iterator<InvChildEdgeIt, PredT>;

  // static ChildEdgeIteratorType child_edge_begin(NodeRef)
  // static ChildEdgeIteratorType child_edge_end(NodeRef)
  //     Return iterators that point to the beginning and ending of the
  //     edge list for the given callgraph node.
  static ChildEdgeIteratorType child_edge_begin(NodeRef N) {
    auto Rng = llvm::make_range(InvGraphTraits::child_edge_begin(N),
                                InvGraphTraits::child_edge_end(N));
    return llvm::make_filter_range(Rng, PredicateType::value).begin();
  }
  static ChildEdgeIteratorType child_edge_end(NodeRef N) {
    auto Rng = llvm::make_range(InvGraphTraits::child_edge_begin(N),
                                InvGraphTraits::child_edge_end(N));
    return llvm::make_filter_range(Rng, PredicateType::value).end();
  }

  // static NodeRef edge_dest(EdgeRef)
  //     Return the destination node of an edge.
  static NodeRef edge_dest(EdgeRef E) { return InvGraphTraits::edge_dest(E); }

  // typedef ChildIteratorType - Type used to iterate over children in graph,
  //                             dereference to a NodeRef.

protected:
  using EdgeDest = NodeRef (*)(EdgeRef);

public:
  using ChildIteratorType = llvm::mapped_iterator<ChildEdgeIteratorType,
                                                  EdgeDest>;

  // static NodeRef getEntryNode(const GraphType &)
  //    Return the entry node of the graph
  static NodeRef getEntryNode(const Inverse &G) {
    return InvGraphTraits::getEntryNode(G.Graph);
  }

  // static ChildIteratorType child_begin(NodeRef)
  // static ChildIteratorType child_end  (NodeRef)
  //    Return iterators that point to the beginning and ending of the child
  //    node list for the specified node.
  static ChildIteratorType child_begin(NodeRef N) {
    return llvm::map_iterator(child_edge_begin(N), edge_dest);
  }
  static ChildIteratorType child_end(NodeRef N) {
    return llvm::map_iterator(child_edge_end(N), edge_dest);
  }
};

// GraphTraits specializations used to provide filtered views on graphs

// Provides a filtered view over llvm::GraphTraits<GraphType>.
// Pred must be a std::integral_constant wrapping a callable that
// computes a boolean predicate over GraphType::EdgeRef.
// This view has all the nodes of the underlying graph, but only the edges for
// which the predicate is true.
template<typename GraphType, typename Pred>
struct llvm::GraphTraits<EdgeFilteredGraphImpl<GraphType, Pred>>
  : public EdgeFilteredGraphImpl<GraphType, Pred> {};

template<typename GraphType, typename Pred>
struct llvm::GraphTraits<llvm::Inverse<EdgeFilteredGraphImpl<GraphType, Pred>>>
  : public llvm::Inverse<EdgeFilteredGraphImpl<GraphType, Pred>> {};

template<typename NodeT>
using EdgeFilter = bool (*)(const typename llvm::GraphTraits<NodeT>::EdgeRef &);

template<typename NodeT, EdgeFilter<NodeT> F>
using EdgeFilteredGraph = EdgeFilteredGraphImpl<NodeT, ic<decltype(F), F>>;
