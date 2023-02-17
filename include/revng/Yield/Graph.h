#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/BasicBlockID.h"

namespace yield {

namespace detail {

using Coordinate = float;
using Dimension = Coordinate;

struct Point {
  Coordinate X;
  Coordinate Y;

  constexpr Point(Coordinate X = 0, Coordinate Y = 0) : X(X), Y(Y) {}
};

struct Size {
  Dimension W;
  Dimension H;

  constexpr Size(Dimension W = 0, Dimension H = 0) : W(W), H(H) {}
};

struct Node {
  explicit Node(const BasicBlockID &Address = BasicBlockID::invalid(),
                const BasicBlockID &NextAddress = BasicBlockID::invalid(),
                const Point &Center = { 0, 0 },
                const Size &Size = { 0, 0 }) :
    Address(Address), NextAddress(NextAddress), Center(Center), Size(Size) {}

  BasicBlockID Address;
  BasicBlockID NextAddress;
  Point Center;
  Size Size;
};

enum class EdgeStatus { Unrouted, Routed, Hidden };
enum class EdgeType { Unconditional, Call, Taken, Refused };

struct Edge {
  EdgeStatus Status = EdgeStatus::Unrouted;
  EdgeType Type = EdgeType::Unconditional;

  std::vector<Point> Path = {};
};

} // namespace detail

using Node = MutableEdgeNode<detail::Node, detail::Edge, false>;

class Graph : public GenericGraph<Node, 16, true> {
public:
  using GenericGraph<Node, 16, true>::GenericGraph;

public:
  using Coordinate = detail::Coordinate;
  using Dimension = detail::Dimension;
  using Point = detail::Point;
  using Size = detail::Size;
  using EdgeStatus = detail::EdgeStatus;
  using EdgeType = detail::EdgeType;
};

} // namespace yield
