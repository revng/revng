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

} // namespace yield
