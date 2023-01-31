/// \file SVG.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/FormatVariadic.h"

#include "revng/Model/Binary.h"
#include "revng/PTML/Tag.h"
#include "revng/Support/GraphAlgorithms.h"
#include "revng/Yield/CallGraphs/CallGraphSlices.h"
#include "revng/Yield/ControlFlow/Configuration.h"
#include "revng/Yield/ControlFlow/Extraction.h"
#include "revng/Yield/ControlFlow/NodeSizeCalculation.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"
#include "revng/Yield/Graph.h"
#include "revng/Yield/PTML.h"
#include "revng/Yield/SVG.h"
#include "revng/Yield/Support/SugiyamaStyleGraphLayout.h"

using ptml::Tag;

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

// clang-format off
template <uintmax_t Numerator = 8, uintmax_t Denominator = 10>
static std::string cubicBend(const yield::Graph::Point &From,
                             const yield::Graph::Point &To,
                             bool VerticalCurves,
                             std::ratio<Numerator, Denominator> &&Bend = {}) {
  // clang-format on

  using Coordinate = yield::Graph::Coordinate;
  constexpr Coordinate Factor = Coordinate(Numerator) / Denominator;
  Coordinate XModifier = Factor * (To.X - From.X);
  Coordinate YModifier = Factor * (To.Y - From.Y);

  if (VerticalCurves)
    XModifier = 0;
  else
    YModifier = 0;

  return llvm::formatv("C {0} {1} {2} {3} {4} {5} ",
                       From.X + XModifier,
                       -From.Y - YModifier,
                       To.X - XModifier,
                       -To.Y + YModifier,
                       To.X,
                       -To.Y);
}

static std::string edge(const std::vector<yield::Graph::Point> &Path,
                        const yield::Graph::EdgeType &Type,
                        bool UseOrthogonalBends = true,
                        bool UseVerticalCurves = false) {
  std::string Points;

  revng_assert(!Path.empty());
  const auto &First = Path.front();
  Points += llvm::formatv("M {0} {1} ", First.X, -First.Y);

  if (UseOrthogonalBends) {
    for (size_t Index = 1; Index < Path.size(); ++Index)
      Points += llvm::formatv("L {0} {1} ", Path[Index].X, -Path[Index].Y);
  } else {
    revng_assert(Path.size() >= 2);
    for (auto Iter = Path.begin(); Iter != std::prev(Path.end()); ++Iter)
      Points += cubicBend(*Iter, *std::next(Iter), UseVerticalCurves);
  }

  revng_assert(!Points.empty());
  revng_assert(Points.back() == ' ');
  Points.pop_back(); // Remove an extra space at the end.

  std::string Marker = llvm::formatv("url(#{0}-arrow-head)",
                                     edgeTypeAsString(Type));
  return Tag("path")
    .addAttribute("class", std::string(edgeTypeAsString(Type)) += "-edge")
    .addAttribute("d", std::move(Points))
    .addAttribute("marker-end", std::move(Marker))
    .addAttribute("fill", "none")
    .serialize();
}

static std::string node(const yield::Node *Node,
                        std::string &&Content,
                        const yield::cfg::Configuration &Configuration) {
  yield::Graph::Size HalfSize{ Node->Size.W / 2, Node->Size.H / 2 };
  yield::Graph::Point TopLeft{ Node->Center.X - HalfSize.W,
                               -Node->Center.Y - HalfSize.H };

  Tag Body("body", std::move(Content));
  Body.addAttribute("xmlns", R"("http://www.w3.org/1999/xhtml")");

  Tag Text("foreignObject", Body.serialize());
  Text.addAttribute("class", ::tags::NodeContents)
    .addAttribute("x", std::to_string(TopLeft.X))
    .addAttribute("y", std::to_string(TopLeft.Y))
    .addAttribute("width", std::to_string(Node->Size.W))
    .addAttribute("height", std::to_string(Node->Size.H));

  Tag Border("rect");
  Border.addAttribute("class", ::tags::NodeBody)
    .addAttribute("x", std::to_string(TopLeft.X))
    .addAttribute("y", std::to_string(TopLeft.Y))
    .addAttribute("rx", std::to_string(Configuration.NodeCornerRoundingFactor))
    .addAttribute("ry", std::to_string(Configuration.NodeCornerRoundingFactor))
    .addAttribute("width", std::to_string(Node->Size.W))
    .addAttribute("height", std::to_string(Node->Size.H));

  return Text.serialize() + Border.serialize();
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

  // Ensure every node fits.
  Viewbox Result = makeViewbox(*Graph.nodes().begin());
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

/// A really simple arrow head marker generator.
///
/// \param Name: the id of the marker as refered to by the objects using it.
/// \param Size: the size of the marker. It sets both width and height to force
/// the marker to be square-shaped.
/// \param Concave: the size of the concave at the rear side of the arrow.
/// \param Shift: the position of the arrow "origin". It is set to 0 by default
/// (arrow origin the same as its tip), positive values shift arrow back,
/// leaving some space between the tip and its target, negative values shift it
/// closer to the target possibly causing an overlap.
static std::string
arrowHead(llvm::StringRef Name, float Size, float Concave, float Shift = 0) {
  std::string Points = llvm::formatv("{0}, {1} {3}, {2} {0}, {0} {1}, {2}",
                                     "0",
                                     std::to_string(Size),
                                     std::to_string(Size / 2),
                                     std::to_string(Concave));

  return Tag("marker",
             Tag("polygon")
               .addAttribute("points", std::move(Points))
               .serialize())
    .addAttribute("id", Name)
    .addAttribute("markerWidth", std::to_string(Size))
    .addAttribute("markerHeight", std::to_string(Size))
    .addAttribute("refX", std::to_string(Size - Shift))
    .addAttribute("refY", std::to_string(Size / 2))
    .addAttribute("refY", std::to_string(Size / 2))
    .addAttribute("orient", "auto")
    .serialize();
}

static std::string
duplicateArrowHeadsImpl(float Size, float Dip, float Shift = 0) {
  return arrowHead(tags::UnconditionalArrowHead, Size, Dip, Shift)
         + arrowHead(tags::CallArrowHead, Size, Dip, Shift)
         + arrowHead(tags::TakenArrowHead, Size, Dip, Shift)
         + arrowHead(tags::RefusedArrowHead, Size, Dip, Shift);
}

static std::string
defaultArrowHeads(const yield::cfg::Configuration &Configuration) {
  if (Configuration.UseOrthogonalBends == true)
    return duplicateArrowHeadsImpl(8, 3, 0);
  else
    return duplicateArrowHeadsImpl(8, 3, 2);
}

template<typename CallableType>
concept NodeExporter = requires(CallableType &&Callable,
                                const yield::Graph::Node &Node) {
  { Callable(Node) } -> convertible_to<std::string>;
};

constexpr bool isVertical(yield::sugiyama::LayoutOrientation Orientation) {
  return Orientation == yield::sugiyama::LayoutOrientation::TopToBottom
         || Orientation == yield::sugiyama::LayoutOrientation::BottomToTop;
}

template<bool ShouldEmitEmptyNodes>
static std::string exportGraph(const yield::Graph &Graph,
                               const yield::cfg::Configuration &Configuration,
                               yield::sugiyama::LayoutOrientation Orientation,
                               NodeExporter auto &&NodeContents) {
  std::string Result;

  // Short circuit the execution for an empty graph.
  if (Graph.size() == 0)
    return Result;

  // Export all the edges.
  for (const auto *From : Graph.nodes()) {
    if (ShouldEmitEmptyNodes || From->Address.isValid()) {
      for (const auto [To, Edge] : From->successor_edges()) {
        if (ShouldEmitEmptyNodes || To->Address.isValid()) {
          revng_assert(Edge != nullptr);
          revng_assert(Edge->Status != yield::Graph::EdgeStatus::Unrouted);
          Result += edge(Edge->Path,
                         Edge->Type,
                         Configuration.UseOrthogonalBends,
                         isVertical(Orientation));
        }
      }
    }
  }

  // Export all the nodes.
  for (const auto *Node : Graph.nodes())
    if (ShouldEmitEmptyNodes || Node->Address.isValid())
      Result += node(Node, NodeContents(*Node), Configuration);

  Viewbox Box = calculateViewbox(Graph);
  std::string SerializedBox = llvm::formatv("{0} {1} {2} {3}",
                                            Box.TopLeft.X,
                                            Box.TopLeft.Y,
                                            Box.BottomRight.X - Box.TopLeft.X,
                                            Box.BottomRight.Y - Box.TopLeft.Y);

  Tag ArrowHeads("defs", defaultArrowHeads(Configuration));
  return Tag("svg", ArrowHeads.serialize() + std::move(Result))
    .addAttribute("xmlns", R"("http://www.w3.org/2000/svg")")
    .addAttribute("viewbox", std::move(SerializedBox))
    .addAttribute("width", std::to_string(Box.BottomRight.X - Box.TopLeft.X))
    .addAttribute("height", std::to_string(Box.BottomRight.Y - Box.TopLeft.Y))
    .serialize();
}

std::string
yield::svg::controlFlowGraph(const yield::Function &InternalFunction,
                             const model::Binary &Binary) {
  constexpr auto Configuration = cfg::Configuration::getDefault();

  yield::Graph Graph = cfg::extractFromInternal(InternalFunction,
                                                Binary,
                                                Configuration);

  cfg::calculateNodeSizes(Graph, InternalFunction, Binary, Configuration);

  constexpr auto Orientation = yield::sugiyama::LayoutOrientation::TopToBottom;
  sugiyama::layout(Graph, Configuration, Orientation);

  auto Content = [&](const yield::Graph::Node &Node) {
    if (Node.Address.isValid())
      return yield::ptml::controlFlowNode(Node.Address,
                                          InternalFunction,
                                          Binary);
    else
      return std::string{};
  };
  return exportGraph<true>(Graph, Configuration, Orientation, Content);
}

struct LabelNodeHelper {
  const model::Binary &Binary;
  const yield::cfg::Configuration Configuration;
  std::optional<MetaAddress> RootNodeLocation = std::nullopt;

  void computeSizes(yield::Graph &Graph) {
    for (auto *Node : Graph.nodes()) {
      if (Node->Address.isValid()) {
        // A normal node
        auto FunctionIterator = Binary.Functions().find(Node->Address);
        revng_assert(FunctionIterator != Binary.Functions().end());

        size_t NameLength = FunctionIterator->name().size();
        revng_assert(NameLength != 0);

        Node->Size = yield::Graph::Size{
          NameLength * Configuration.LabelFontSize
            * Configuration.HorizontalFontFactor,
          1 * Configuration.LabelFontSize * Configuration.VerticalFontFactor
        };
      } else {
        // An entry node.
        Node->Size = yield::Graph::Size{ 30, 30 };
      }

      Node->Size.W += Configuration.InternalNodeMarginSize * 2;
      Node->Size.H += Configuration.InternalNodeMarginSize * 2;
    }
  }

  std::string operator()(const yield::Graph::Node &Node) const {
    revng_assert(Node.Address.isValid());
    if (Node.NextAddress.isValid()) {
      revng_assert(Node.Address == Node.NextAddress);
      return yield::ptml::shallowFunctionLink(Node.NextAddress, Binary);
    }

    if (!RootNodeLocation.has_value() || *RootNodeLocation == Node.Address)
      return yield::ptml::functionNameDefinition(Node.Address, Binary);
    else
      return yield::ptml::functionLink(Node.Address, Binary);
  }
};

using CrossRelations = yield::crossrelations::CrossRelations;
std::string yield::svg::callGraph(const CrossRelations &Relations,
                                  const model::Binary &Binary) {
  // TODO: make configuration accessible from outside.
  auto Configuration = cfg::Configuration::getDefault();
  Configuration.UseOrthogonalBends = false;
  constexpr auto Orientation = sugiyama::LayoutOrientation::LeftToRight;
  constexpr auto Ranking = sugiyama::RankingStrategy::BreadthFirstSearch;

  LabelNodeHelper Helper{ Binary, Configuration };

  auto Result = Relations.toYieldGraph();
  auto EntryPoints = entryPoints(&Result);
  revng_assert(!EntryPoints.empty());
  if (EntryPoints.size() > 1) {
    // Add an artificial "root" node to make sure there's a single entry point.
    yield::Graph::Node *Root = Result.addNode();
    for (yield::Graph::Node *Entry : EntryPoints)
      Root->addSuccessor(Entry);
    Result.setEntryNode(Root);
  } else {
    Result.setEntryNode(EntryPoints.front());
  }

  auto InternalGraph = calls::makeCalleeTree(Result);
  Helper.computeSizes(InternalGraph);

  sugiyama::layout(InternalGraph, Configuration, Orientation, Ranking, true);
  return exportGraph<false>(InternalGraph, Configuration, Orientation, Helper);
}

static auto flipPoint(yield::Graph::Point const &Point) {
  return yield::Graph::Point{ -Point.X, -Point.Y };
};
static auto
calculateDelta(yield::Graph::Point const &LHS, yield::Graph::Point const &RHS) {
  return yield::Graph::Point{ RHS.X - LHS.X, RHS.Y - LHS.Y };
}
static auto translatePoint(yield::Graph::Point const &Point,
                           yield::Graph::Point const &Delta) {
  return yield::Graph::Point{ Point.X + Delta.X, Point.Y + Delta.Y };
}
static auto convertPoint(yield::Graph::Point const &Point,
                         yield::Graph::Point const &Delta) {
  return translatePoint(flipPoint(Point), Delta);
}

static yield::Graph combineHalvesHelper(const MetaAddress &SlicePoint,
                                        yield::Graph &&ForwardsSlice,
                                        yield::Graph &&BackwardsSlice) {
  revng_assert(ForwardsSlice.size() != 0 && BackwardsSlice.size() != 0);

  auto IsSlicePoint = [&SlicePoint](const auto *Node) {
    return Node->Address == SlicePoint;
  };

  auto ForwardsIterator = llvm::find_if(ForwardsSlice.nodes(), IsSlicePoint);
  revng_assert(ForwardsIterator != ForwardsSlice.nodes().end());
  auto *ForwardsSlicePoint = *ForwardsIterator;
  revng_assert(ForwardsSlicePoint != nullptr);

  auto BackwardsIterator = llvm::find_if(BackwardsSlice.nodes(), IsSlicePoint);
  revng_assert(BackwardsIterator != BackwardsSlice.nodes().end());
  auto *BackwardsSlicePoint = *BackwardsIterator;
  revng_assert(BackwardsSlicePoint != nullptr);

  // Find the distance all the nodes of one of the graphs need to be shifted so
  // that the `SlicePoint`s overlap.
  auto Delta = calculateDelta(flipPoint((*BackwardsIterator)->Center),
                              (*ForwardsIterator)->Center);

  // Ready the backwards part of the graph
  for (auto *From : BackwardsSlice.nodes()) {
    From->Center = convertPoint(From->Center, Delta);
    for (auto [Neighbor, Label] : From->successor_edges())
      for (auto &Point : Label->Path)
        Point = convertPoint(Point, Delta);
  }

  // Define a map for faster node lookup.
  llvm::DenseMap<yield::Graph::Node *, yield::Graph::Node *> Lookup;
  auto AccessLookup = [&Lookup](yield::Graph::Node *Key) {
    auto Iterator = Lookup.find(Key);
    revng_assert(Iterator != Lookup.end() && Iterator->second != nullptr);
    return Iterator->second;
  };

  // Move the nodes from the backwards slice into the forwards one.
  for (auto *Node : BackwardsSlice.nodes()) {
    revng_assert(Node != nullptr);
    if (Node != BackwardsSlicePoint) {
      auto NewNode = ForwardsSlice.addNode(Node->moveData());
      auto [Iterator, Success] = Lookup.try_emplace(Node, NewNode);
      revng_assert(Success == true);
    } else {
      auto [Iterator, Success] = Lookup.try_emplace(BackwardsSlicePoint,
                                                    ForwardsSlicePoint);
      revng_assert(Success == true);
    }
  }

  // Move all the edges while also inverting their direction.
  for (auto *From : BackwardsSlice.nodes()) {
    for (auto [To, Label] : From->successor_edges()) {
      std::reverse(Label->Path.begin(), Label->Path.end());
      AccessLookup(To)->addSuccessor(AccessLookup(From), std::move(*Label));
    }
  }

  return std::move(ForwardsSlice);
}

std::string yield::svg::callGraphSlice(const MetaAddress &SlicePoint,
                                       const CrossRelations &Relations,
                                       const model::Binary &Binary) {
  // TODO: make configuration accessible from outside.
  auto Configuration = cfg::Configuration::getDefault();
  Configuration.UseOrthogonalBends = false;
  constexpr auto Orientation = sugiyama::LayoutOrientation::LeftToRight;
  constexpr auto Ranking = sugiyama::RankingStrategy::BreadthFirstSearch;

  LabelNodeHelper Helper{ Binary, Configuration, SlicePoint };

  // Ready the forwards facing part of the slice
  auto ForwardsGraph = calls::makeCalleeTree(Relations.toYieldGraph(),
                                             SlicePoint);
  for (auto *From : ForwardsGraph.nodes())
    for (auto [To, Label] : From->successor_edges())
      Label->Type = yield::Graph::EdgeType::Taken;
  Helper.computeSizes(ForwardsGraph);
  sugiyama::layout(ForwardsGraph, Configuration, Orientation, Ranking, true);

  // Ready the backwards facing part of the slice
  auto BackwardsGraph = calls::makeCallerTree(Relations.toYieldGraph(),
                                              SlicePoint);
  for (auto *From : BackwardsGraph.nodes())
    for (auto [To, Label] : From->successor_edges())
      Label->Type = yield::Graph::EdgeType::Refused;
  Helper.computeSizes(BackwardsGraph);
  sugiyama::layout(BackwardsGraph, Configuration, Orientation, Ranking, true);

  // Consume the halves to produce a combined graph and export it.
  auto CombinedGraph = combineHalvesHelper(SlicePoint,
                                           std::move(ForwardsGraph),
                                           std::move(BackwardsGraph));
  return exportGraph<false>(CombinedGraph, Configuration, Orientation, Helper);
}
