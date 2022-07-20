/// \file SVG.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/FormatVariadic.h"

#include "revng/Model/Binary.h"
#include "revng/Yield/ControlFlow/Configuration.h"
#include "revng/Yield/ControlFlow/Extraction.h"
#include "revng/Yield/ControlFlow/NodeSizeCalculation.h"
#include "revng/Yield/Graph.h"
#include "revng/Yield/PTML.h"
#include "revng/Yield/SVG.h"
#include "revng/Yield/Support/SugiyamaStyleGraphLayout.h"

namespace tags {

static constexpr auto UnconditionalEdge = "unconditional";
static constexpr auto CallEdge = "call";
static constexpr auto TakenEdge = "taken";
static constexpr auto RefusedEdge = "refused";

static constexpr auto NodeBody = "node-body";
static constexpr auto NodeContents = "node-contents";

static constexpr auto UnconditionalArrowHead = "unconditional-arrow-head";
static constexpr auto CallArrowHead = "call-arrow-head";
static constexpr auto TakenArrowHead = "taken-arrow-head";
static constexpr auto RefusedArrowHead = "refused-arrow-head";

} // namespace tags

namespace templates {

constexpr auto *Point = "M {0} {1} ";
constexpr auto *Line = "L {0} {1} ";
constexpr auto *BezierCubic = "C {0} {1} {2} {3} {4} {5} ";

constexpr auto *Edge = R"Edge(<path
  class='{0}-edge'
  {1}
  marker-end="url(#{0}-arrow-head)"
  fill="none" />
)Edge";

constexpr auto *Rect = R"(<rect
  class="{0}"
  x="{1}"
  y="{2}"
  rx="{3}"
  ry="{3}"
  width="{4}"
  height="{5}" />)";

constexpr auto *ForeignObject = R"(<foreignObject
  class="{0}"
  x="{1}"
  y="{2}"
  width="{3}"
  height="{4}"><body
  xmlns="http://www.w3.org/1999/xhtml">{5}</body></foreignObject>
)";

static constexpr auto ArrowHead = R"(<marker
  id="{0}"
  markerWidth="{1}"
  markerHeight="{1}"
  refX="{1}"
  refY="{2}"
  orient="{5}"><polygon
  points="{4}, {1} {3}, {2} {4}, {4} {1}, {2}" /></marker>
)";

static constexpr auto Graph = R"(<svg
  xmlns="http://www.w3.org/2000/svg"
  viewbox="{0} {1} {2} {3}"
  width="{2}"
  height="{3}"><defs>{4}</defs>{5}</svg>
)";

} // namespace templates

static std::string_view edgeTypeAsString(yield::Graph::EdgeType Type) {
  switch (Type) {
  case yield::Graph::EdgeType::Unconditional:
    return tags::UnconditionalEdge;
  case yield::Graph::EdgeType::Call:
    return tags::CallEdge;
  case yield::Graph::EdgeType::Taken:
    return tags::TakenEdge;
  case yield::Graph::EdgeType::Refused:
    return tags::RefusedEdge;
  default:
    revng_abort("Unknown edge type");
  }
}

static std::string edge(const std::vector<yield::Graph::Point> &Path,
                        const yield::Graph::EdgeType &Type,
                        bool UseOrthogonalBends) {
  std::string Points = "d=\"";

  revng_assert(!Path.empty());
  const auto &First = Path.front();
  Points += llvm::formatv(templates::Point, First.X, -First.Y);

  if (UseOrthogonalBends) {
    for (size_t Index = 1; Index < Path.size(); ++Index)
      Points += llvm::formatv(templates::Line, Path[Index].X, -Path[Index].Y);
  } else {
    revng_assert(Path.size() == 2);
    const auto &Second = Path.back();

    static constexpr yield::Graph::Coordinate BendFactor = 0.8;

    yield::Graph::Coordinate XDistance = Second.X - First.X;
    Points += llvm::formatv(templates::BezierCubic,
                            First.X + XDistance * BendFactor,
                            -First.Y,
                            Second.X - XDistance * BendFactor,
                            -Second.Y,
                            Second.X,
                            -Second.Y);
  }

  revng_assert(!Points.empty());
  revng_assert(Points.back() == ' ');
  Points.pop_back(); // Remove an extra space at the end.
  Points += '"';

  return llvm::formatv(templates::Edge, edgeTypeAsString(Type), Points);
}

static std::string content(const yield::Node *Node,
                           const yield::Function &Function,
                           const model::Binary &Binary) {
  revng_assert(Node != nullptr);

  if (Node->Address.isValid()) {
    // A normal node.
    return yield::ptml::controlFlowNode(Node->Address, Function, Binary);
  } else {
    // The entry/exit/error node.
    return "";
  }
}

static std::string node(const yield::Node *Node,
                        const yield::Function &Function,
                        const model::Binary &Binary,
                        const yield::cfg::Configuration &Configuration) {
  yield::Graph::Size HalfSize{ Node->Size.W / 2, Node->Size.H / 2 };
  yield::Graph::Point TopLeft{ Node->Center.X - HalfSize.W,
                               -Node->Center.Y - HalfSize.H };

  auto Text = llvm::formatv(templates::ForeignObject,
                            tags::NodeContents,
                            TopLeft.X,
                            TopLeft.Y,
                            Node->Size.W,
                            Node->Size.H,
                            content(Node, Function, Binary));

  auto Body = llvm::formatv(templates::Rect,
                            tags::NodeBody,
                            TopLeft.X,
                            TopLeft.Y,
                            Configuration.NodeCornerRoundingFactor,
                            Node->Size.W,
                            Node->Size.H);

  return llvm::formatv("{0}", std::move(Text) + std::move(Body));
}

struct Viewbox {
  yield::Graph::Point TopLeft = { -1, -1 };
  yield::Graph::Point BottomRight = { +1, +1 };
};

static Viewbox makeViewbox(const yield::Node *Node) {
  yield::Graph::Size HalfSize{ Node->Size.W / 2, Node->Size.H / 2 };
  yield::Graph::Point TopLeft{ Node->Center.X - HalfSize.W,
                               -Node->Center.Y - HalfSize.H };
  yield::Graph::Point BottomRight{ Node->Center.X + HalfSize.W,
                                   -Node->Center.Y + HalfSize.H };
  return Viewbox{ .TopLeft = std::move(TopLeft),
                  .BottomRight = std::move(BottomRight) };
}

static void expandViewbox(Viewbox &LHS, const Viewbox &RHS) {
  if (RHS.TopLeft.X < LHS.TopLeft.X)
    LHS.TopLeft.X = RHS.TopLeft.X;
  if (RHS.TopLeft.Y < LHS.TopLeft.Y)
    LHS.TopLeft.Y = RHS.TopLeft.Y;
  if (RHS.BottomRight.X > LHS.BottomRight.X)
    LHS.BottomRight.X = RHS.BottomRight.X;
  if (RHS.BottomRight.Y > LHS.BottomRight.Y)
    LHS.BottomRight.Y = RHS.BottomRight.Y;
}

static void expandViewbox(Viewbox &Box, const yield::Graph::Point &Point) {
  if (Box.TopLeft.X > Point.X)
    Box.TopLeft.X = Point.X;
  if (Box.TopLeft.Y > -Point.Y)
    Box.TopLeft.Y = -Point.Y;
  if (Box.BottomRight.X < Point.X)
    Box.BottomRight.X = Point.X;
  if (Box.BottomRight.Y < -Point.Y)
    Box.BottomRight.Y = -Point.Y;
}

static Viewbox calculateViewbox(const yield::Graph &Graph) {
  revng_assert(Graph.size() != 0);
  revng_assert(Graph.getEntryNode() != nullptr);

  // Ensure every node fits.
  Viewbox Result = makeViewbox(Graph.getEntryNode());
  for (const auto *Node : Graph.nodes())
    expandViewbox(Result, makeViewbox(Node));

  // Ensure every edge point fits.
  for (const auto *From : Graph.nodes())
    for (const auto [To, Label] : From->successor_edges())
      for (const auto &Point : Label->Path)
        expandViewbox(Result, Point);

  // Add some extra padding for a good measure.
  Result.TopLeft.X -= 50;
  Result.TopLeft.Y -= 50;
  Result.BottomRight.X += 50;
  Result.BottomRight.Y += 50;

  return Result;
}

static std::string arrowHead(llvm::StringRef Name, float Size, float Dip) {
  return llvm::formatv(templates::ArrowHead,
                       Name,
                       std::to_string(Size),
                       std::to_string(Size / 2),
                       std::to_string(Dip),
                       "0",
                       "auto");
}

static std::string defaultArrowHeads() {
  return arrowHead(tags::UnconditionalArrowHead, 8, 3)
         + arrowHead(tags::CallArrowHead, 8, 3)
         + arrowHead(tags::TakenArrowHead, 8, 3)
         + arrowHead(tags::RefusedArrowHead, 8, 3);
}

static std::string exportCFG(const yield::Graph &Graph,
                             const yield::Function &Function,
                             const model::Binary &Binary,
                             const yield::cfg::Configuration &Configuration) {
  std::string Result;

  // Export all the edges.
  for (const auto *From : Graph.nodes()) {
    for (const auto [To, Edge] : From->successor_edges()) {
      revng_assert(Edge != nullptr);
      // TODO: remove this if after the layouter is functional!
      if (!Edge->Path.empty()) {
        revng_assert(Edge->Status != yield::Graph::EdgeStatus::Unrouted);
        Result += edge(Edge->Path,
                       Edge->Type,
                       Configuration.UseOrthogonalBends);
      }
    }
  }

  // Export all the nodes.
  for (const auto *Node : Graph.nodes())
    Result += node(Node, Function, Binary, Configuration);

  Viewbox Box = calculateViewbox(Graph);
  return llvm::formatv(templates::Graph,
                       Box.TopLeft.X,
                       Box.TopLeft.Y,
                       Box.BottomRight.X - Box.TopLeft.X,
                       Box.BottomRight.Y - Box.TopLeft.Y,
                       defaultArrowHeads(),
                       std::move(Result));
}

std::string yield::svg::controlFlow(const yield::Function &InternalFunction,
                                    const model::Binary &Binary) {
  constexpr auto Configuration = cfg::Configuration::getDefault();

  yield::Graph ControlFlowGraph = cfg::extractFromInternal(InternalFunction,
                                                           Binary,
                                                           Configuration);

  cfg::calculateNodeSizes(ControlFlowGraph,
                          InternalFunction,
                          Binary,
                          Configuration);

  sugiyama::layout(ControlFlowGraph, Configuration);

  return exportCFG(ControlFlowGraph, InternalFunction, Binary, Configuration);
}
