#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/MFP/MFP.h"

namespace MFP {

template<MonotoneFrameworkInstance MFI>
class Graph {
public:
  using Label = typename MFI::Label;
  using LatticeElement = typename MFI::LatticeElement;
  using MFIResults = std::map<Label, MFPResult<LatticeElement>>;

private:
  MFI::GraphType UnderlyingGraph;
  const MFIResults &Results;

public:
  Graph(MFI::GraphType UnderlyingGraph, const MFIResults &Results) :
    UnderlyingGraph(UnderlyingGraph), Results(Results) {}

public:
  auto underlying() const { return UnderlyingGraph; }
  const auto &results() const { return Results; }
};

template<typename T>
void dump(llvm::raw_ostream &Stream, unsigned Indent, const T &Element);

template<typename T>
concept SerializableLatticeElement = requires(llvm::raw_ostream &Stream,
                                              const T &Element) {
  { MFP::dump(Stream, 0, Element) } -> std::same_as<void>;
};

} // namespace MFP

/// \note This implementation of GraphTraits forwards everything 1-to-1
template<MFP::MonotoneFrameworkInstance MFI>
struct llvm::GraphTraits<MFP::Graph<MFI> *> {
  using GraphType = MFP::Graph<MFI> *;
  using UnderlyingGraphTraits = llvm::GraphTraits<typename MFI::GraphType>;
  using NodeRef = typename UnderlyingGraphTraits::NodeRef;
  using EdgeRef = typename UnderlyingGraphTraits::EdgeRef;
  using nodes_iterator = typename UnderlyingGraphTraits::nodes_iterator;
  using ChildIteratorType = typename UnderlyingGraphTraits::ChildIteratorType;

  static NodeRef getEntryNode(GraphType G) {
    return UnderlyingGraphTraits::getEntryNode(G->underlying());
  }

  static auto nodes_begin(GraphType G) {
    return UnderlyingGraphTraits::nodes_begin(G->underlying());
  }

  static auto nodes_end(GraphType G) {
    return UnderlyingGraphTraits::nodes_end(G->underlying());
  }

  static size_t size(GraphType G) {
    return UnderlyingGraphTraits::size(G->underlying());
  }

  static auto child_begin(NodeRef N) {
    return UnderlyingGraphTraits::child_begin(N);
  }
  static auto child_end(NodeRef N) {
    return UnderlyingGraphTraits::child_end(N);
  }

  static auto child_edge_begin(NodeRef N) {
    return UnderlyingGraphTraits::child_edge_begin(N);
  }
  static auto child_edge_end(NodeRef N) {
    return UnderlyingGraphTraits::child_edge_end(N);
  }

  static NodeRef *edge_dest(EdgeRef Edge) {
    return UnderlyingGraphTraits::edge_dest(Edge);
  }
};

/// \note This implementation of GraphTraits forwards everything 1-to-1
template<MFP::MonotoneFrameworkInstance MFI>
struct llvm::GraphTraits<const MFP::Graph<MFI> *> {
  using GraphType = const MFP::Graph<MFI> *;
  using UnderlyingGraphTraits = llvm::GraphTraits<typename MFI::GraphType>;
  using NodeRef = const typename UnderlyingGraphTraits::NodeRef;
  using EdgeRef = typename UnderlyingGraphTraits::EdgeRef;
  using nodes_iterator = typename UnderlyingGraphTraits::nodes_iterator;
  using ChildIteratorType = typename UnderlyingGraphTraits::ChildIteratorType;

  static NodeRef getEntryNode(GraphType G) {
    return UnderlyingGraphTraits::getEntryNode(G->underlying());
  }

  static auto nodes_begin(GraphType G) {
    return UnderlyingGraphTraits::nodes_begin(G->underlying());
  }

  static auto nodes_end(GraphType G) {
    return UnderlyingGraphTraits::nodes_end(G->underlying());
  }

  static size_t size(GraphType G) {
    return UnderlyingGraphTraits::size(G->underlying());
  }

  static auto child_begin(NodeRef N) {
    return UnderlyingGraphTraits::child_begin(N);
  }
  static auto child_end(NodeRef N) {
    return UnderlyingGraphTraits::child_end(N);
  }

  static auto child_edge_begin(NodeRef N) {
    return UnderlyingGraphTraits::child_edge_begin(N);
  }
  static auto child_edge_end(NodeRef N) {
    return UnderlyingGraphTraits::child_edge_end(N);
  }

  static NodeRef *edge_dest(EdgeRef Edge) {
    return UnderlyingGraphTraits::edge_dest(Edge);
  }
};
