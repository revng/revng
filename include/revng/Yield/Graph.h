#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/BasicBlockID.h"
#include "revng/Yield/Support/GraphLayout/Traits.h"

namespace yield {

namespace detail {

struct Node {
  explicit Node(const BasicBlockID &Address = BasicBlockID::invalid(),
                const BasicBlockID &NextAddress = BasicBlockID::invalid(),
                const layout::Point &Center = { 0, 0 },
                const layout::Size &Size = { 0, 0 }) :
    Address(Address), NextAddress(NextAddress), Center(Center), Size(Size) {}

  BasicBlockID Address;
  BasicBlockID NextAddress;
  layout::Point Center;
  layout::Size Size;
};

enum class EdgeStatus { Unrouted, Routed, Hidden };
enum class EdgeType { Unconditional, Call, Taken, Refused };

struct Edge {
  EdgeStatus Status = EdgeStatus::Unrouted;
  EdgeType Type = EdgeType::Unconditional;

  std::vector<layout::Point> Path = {};
};

} // namespace detail

using Node = MutableEdgeNode<detail::Node, detail::Edge, false>;

class Graph : public GenericGraph<Node, 16, true> {
public:
  using GenericGraph<Node, 16, true>::GenericGraph;

public:
  using EdgeStatus = detail::EdgeStatus;
  using EdgeType = detail::EdgeType;
};

namespace layout {

template<>
struct LayoutableGraphTraits<Graph *> {
  static_assert(SpecializationOfGenericGraph<Graph>);
  using LLVMTrait = llvm::GraphTraits<Graph *>;

  static const Size &getNodeSize(typename LLVMTrait::NodeRef Node) {
    return Node->Size;
  }

  static void setNodePosition(typename LLVMTrait::NodeRef Node, Point &&Point) {
    Node->Center = std::move(Point);
  }
  static void setEdgePath(typename LLVMTrait::EdgeRef Edge, Path &&Path) {
    Edge.Label->Path.reserve(Path.size());
    for (layout::Point &Point : Path)
      Edge.Label->Path.emplace_back(std::move(Point));
  }
};

} // namespace layout

} // namespace yield
