/// \file SVG.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <unordered_map>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/FormatVariadic.h"

#include "revng/GraphLayout/SugiyamaStyle/Compute.h"
#include "revng/Model/Binary.h"
#include "revng/PTML/Tag.h"
#include "revng/Support/GraphAlgorithms.h"
#include "revng/Yield/CallGraphs/CallGraphSlices.h"
#include "revng/Yield/ControlFlow/Configuration.h"
#include "revng/Yield/ControlFlow/Extraction.h"
#include "revng/Yield/ControlFlow/NodeSizeCalculation.h"
#include "revng/Yield/CrossRelations/CrossRelations.h"
#include "revng/Yield/PTML.h"
#include "revng/Yield/SVG.h"

using ptml::PTMLBuilder;
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

static std::string_view edgeTypeAsString(const yield::cfg::Edge &Edge) {
  switch (Edge.Type) {
  case yield::cfg::EdgeType::Unconditional:
    return tags::UnconditionalEdge;
  case yield::cfg::EdgeType::Call:
    return tags::CallEdge;
  case yield::cfg::EdgeType::Taken:
    return tags::TakenEdge;
  case yield::cfg::EdgeType::Refused:
    return tags::RefusedEdge;
  default:
    revng_abort("Unknown edge type");
  }
}

static std::string_view edgeTypeAsString(const yield::calls::Edge &Edge) {
  // TODO: we might want to use separate set of tags for call graphs.
  return Edge.IsBackwards ? tags::RefusedEdge : tags::TakenEdge;
}

template<uintmax_t Numerator = 8, uintmax_t Denominator = 10>
static std::string cubicBend(const yield::layout::Point &From,
                             const yield::layout::Point &To,
                             bool VerticalCurves,
                             std::ratio<Numerator, Denominator> &&Bend = {}) {
  using Coordinate = yield::layout::Coordinate;
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

static std::string edge(const PTMLBuilder &ThePTMLBuilder,
                        const yield::layout::Path &Path,
                        const std::string_view Type,
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

  std::string Marker = llvm::formatv("url(#{0}-arrow-head)", Type);
  return ThePTMLBuilder.getTag("path")
    .addAttribute("class", std::string(Type) += "-edge")
    .addAttribute("d", std::move(Points))
    .addAttribute("marker-end", std::move(Marker))
    .addAttribute("fill", "none")
    .serialize();
}

template<typename NodeData, typename EdgeData = Empty>
std::string node(const PTMLBuilder &ThePTMLBuilder,
                 const yield::layout::OutputNode<NodeData, EdgeData> *Node,
                 std::string &&Content,
                 const yield::cfg::Configuration &Configuration) {
  yield::layout::Size HalfSize{ Node->Size.W / 2, Node->Size.H / 2 };
  yield::layout::Point TopLeft{ Node->Center.X - HalfSize.W,
                                -Node->Center.Y - HalfSize.H };

  Tag Body = ThePTMLBuilder.getTag("body", std::move(Content));
  Body.addAttribute("xmlns", R"(http://www.w3.org/1999/xhtml)");

  Tag Text = ThePTMLBuilder.getTag("foreignObject", Body.serialize());
  Text.addAttribute("class", ::tags::NodeContents)
    .addAttribute("x", std::to_string(TopLeft.X))
    .addAttribute("y", std::to_string(TopLeft.Y))
    .addAttribute("width", std::to_string(Node->Size.W))
    .addAttribute("height", std::to_string(Node->Size.H));

  Tag Border = ThePTMLBuilder.getTag("rect");
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
  yield::layout::Point TopLeft = { -1, -1 };
  yield::layout::Point BottomRight = { +1, +1 };
};

template<typename NodeData, typename EdgeData = Empty>
static Viewbox
makeViewbox(const yield::layout::OutputNode<NodeData, EdgeData> *Node) {
  yield::layout::Size HalfSize{ Node->Size.W / 2, Node->Size.H / 2 };
  yield::layout::Point TopLeft{ Node->Center.X - HalfSize.W,
                                -Node->Center.Y - HalfSize.H };
  yield::layout::Point BottomRight{ Node->Center.X + HalfSize.W,
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

static void expandViewbox(Viewbox &Box, const yield::layout::Point &Point) {
  if (Box.TopLeft.X > Point.X)
    Box.TopLeft.X = Point.X;
  if (Box.TopLeft.Y > -Point.Y)
    Box.TopLeft.Y = -Point.Y;
  if (Box.BottomRight.X < Point.X)
    Box.BottomRight.X = Point.X;
  if (Box.BottomRight.Y < -Point.Y)
    Box.BottomRight.Y = -Point.Y;
}

template<SpecializationOf<yield::layout::OutputGraph> GraphType>
Viewbox calculateViewbox(const GraphType &Graph) {
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
/// \param ThePTMLBuilder: PTML builder that should be used to make ptml::Tags.
/// \param Name: the id of the marker as referred to by the objects using it.
/// \param Size: the size of the marker. It sets both width and height to force
/// the marker to be square-shaped.
/// \param Concave: the size of the concave at the rear side of the arrow.
/// \param Shift: the position of the arrow "origin". It is set to 0 by default
/// (arrow origin the same as its tip), positive values shift arrow back,
/// leaving some space between the tip and its target, negative values shift it
/// closer to the target possibly causing an overlap.
static std::string arrowHead(const ::ptml::PTMLBuilder &ThePTMLBuilder,
                             llvm::StringRef Name,
                             float Size,
                             float Concave,
                             float Shift = 0) {
  std::string Points = llvm::formatv("{0}, {1} {3}, {2} {0}, {0} {1}, {2}",
                                     "0",
                                     std::to_string(Size),
                                     std::to_string(Size / 2),
                                     std::to_string(Concave));

  return ThePTMLBuilder
    .getTag("marker",
            ThePTMLBuilder.getTag("polygon")
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
duplicateArrowHeadsImpl(const ::ptml::PTMLBuilder &ThePTMLBuilder,
                        float Size,
                        float Dip,
                        float Shift = 0) {
  return arrowHead(ThePTMLBuilder,
                   tags::UnconditionalArrowHead,
                   Size,
                   Dip,
                   Shift)
         + arrowHead(ThePTMLBuilder, tags::CallArrowHead, Size, Dip, Shift)
         + arrowHead(ThePTMLBuilder, tags::TakenArrowHead, Size, Dip, Shift)
         + arrowHead(ThePTMLBuilder, tags::RefusedArrowHead, Size, Dip, Shift);
}

static std::string
defaultArrowHeads(const ::ptml::PTMLBuilder &ThePTMLBuilder,
                  const yield::cfg::Configuration &Configuration) {
  if (Configuration.UseOrthogonalBends == true)
    return duplicateArrowHeadsImpl(ThePTMLBuilder, 8, 3, 0);
  else
    return duplicateArrowHeadsImpl(ThePTMLBuilder, 8, 3, 2);
}

constexpr bool isVertical(yield::layout::sugiyama::Orientation Orientation) {
  return Orientation == yield::layout::sugiyama::Orientation::TopToBottom
         || Orientation == yield::layout::sugiyama::Orientation::BottomToTop;
}

template<typename CallableType, typename NodeType>
concept NodeExporter = requires(CallableType &&Callable, const NodeType &Node) {
  { Callable(Node) } -> convertible_to<std::string>;
};

template<bool ShouldEmitEmptyNodes,
         SpecializationOf<yield::layout::OutputGraph> PostLayoutGraph,
         NodeExporter<typename PostLayoutGraph::Node> ContentsLambda>
static std::string exportGraph(const PTMLBuilder &ThePTMLBuilder,
                               const PostLayoutGraph &Graph,
                               const yield::cfg::Configuration &Configuration,
                               yield::layout::sugiyama::Orientation Orientation,
                               ContentsLambda &&NodeContents) {
  std::string Result;

  // Short circuit the execution for an empty graph.
  if (Graph.size() == 0)
    return Result;

  // Export all the edges.
  for (const auto *From : Graph.nodes()) {
    if (ShouldEmitEmptyNodes || !From->isEmpty()) {
      for (const auto [To, Edge] : From->successor_edges()) {
        if (ShouldEmitEmptyNodes || !To->isEmpty()) {
          revng_assert(Edge != nullptr);
          Result += edge(ThePTMLBuilder,
                         Edge->Path,
                         edgeTypeAsString(*Edge),
                         Configuration.UseOrthogonalBends,
                         isVertical(Orientation));
        }
      }
    }
  }

  // Export all the nodes.
  for (const auto *Node : Graph.nodes())
    if (ShouldEmitEmptyNodes || !Node->isEmpty())
      Result += node(ThePTMLBuilder, Node, NodeContents(*Node), Configuration);

  Viewbox Box = calculateViewbox(Graph);
  std::string SerializedBox = llvm::formatv("{0} {1} {2} {3}",
                                            Box.TopLeft.X,
                                            Box.TopLeft.Y,
                                            Box.BottomRight.X - Box.TopLeft.X,
                                            Box.BottomRight.Y - Box.TopLeft.Y);

  Tag ArrowHeads = ThePTMLBuilder.getTag("defs",
                                         defaultArrowHeads(ThePTMLBuilder,
                                                           Configuration));
  return ThePTMLBuilder
    .getTag("svg", ArrowHeads.serialize() + std::move(Result))
    .addAttribute("xmlns", R"(http://www.w3.org/2000/svg)")
    .addAttribute("viewbox", std::move(SerializedBox))
    .addAttribute("width", std::to_string(Box.BottomRight.X - Box.TopLeft.X))
    .addAttribute("height", std::to_string(Box.BottomRight.Y - Box.TopLeft.Y))
    .serialize();
}

namespace yield::layout::sugiyama {

/// A helper for invoking sugiyama style layouter with the configuration
/// filled in based on the relevant cfg::Configuration.
///
/// \tparam Node The type of the data attached to each graph node
/// \tparam Edge The type of the data attached to each graph edge
///
/// \param Graph An input graph
/// \param CFG An object describing the desired CFG configuration
/// \param LayoutOrientation The direction of the desired layout
/// \param Ranking The ranking strategy
/// \param UseSimpleTreeOptimization A flag deciding whether simple tree
///        optimization should be used.
///
/// \return The laid out version of the graph corresponding to \ref Graph
template<typename Node, typename Edge = Empty>
inline std::optional<OutputGraph<Node, Edge>>
compute(const InputGraph<Node, Edge> &Graph,
        const cfg::Configuration &CFG,
        Orientation LayoutOrientation = Orientation::TopToBottom,
        RankingStrategy Ranking = RankingStrategy::DisjointDepthFirstSearch,
        bool UseSimpleTreeOptimization = false) {
  return compute(Graph,
                 Configuration{
                   .Ranking = Ranking,
                   .Orientation = LayoutOrientation,
                   .UseOrthogonalBends = CFG.UseOrthogonalBends,
                   .PreserveLinearSegments = CFG.PreserveLinearSegments,
                   .UseSimpleTreeOptimization = UseSimpleTreeOptimization,
                   .VirtualNodeWeight = CFG.VirtualNodeWeight,
                   .NodeMarginSize = CFG.ExternalNodeMarginSize,
                   .EdgeMarginSize = CFG.EdgeMarginSize });
}

} // namespace yield::layout::sugiyama

std::string
yield::svg::controlFlowGraph(const PTMLBuilder &ThePTMLBuilder,
                             const yield::Function &InternalFunction,
                             const model::Binary &Binary) {
  constexpr auto Configuration = cfg::Configuration::getDefault();

  using Pre = cfg::PreLayoutGraph;
  Pre Graph = cfg::extractFromInternal(InternalFunction, Binary, Configuration);

  cfg::calculateNodeSizes(Graph, InternalFunction, Binary, Configuration);

  constexpr auto TopToBottom = layout::sugiyama::Orientation::TopToBottom;

  using Post = std::optional<cfg::PostLayoutGraph>;
  Post Result = layout::sugiyama::compute(Graph, Configuration, TopToBottom);
  revng_assert(Result.has_value());

  auto Content = [&](const yield::cfg::PostLayoutNode &Node) {
    if (!Node.isEmpty())
      return yield::ptml::controlFlowNode(ThePTMLBuilder,
                                          Node.getBasicBlock(),
                                          InternalFunction,
                                          Binary);
    else
      return std::string{};
  };
  return exportGraph<true>(ThePTMLBuilder,
                           *Result,
                           Configuration,
                           TopToBottom,
                           Content);
}

struct LabelNodeHelper {
  const PTMLBuilder &ThePTMLBuilder;
  const model::Binary &Binary;
  const yield::cfg::Configuration Configuration;
  std::optional<std::string_view> RootNodeLocation = std::nullopt;

  void computeSizes(yield::calls::PreLayoutGraph &Graph) {
    for (auto *Node : Graph.nodes()) {
      if (!Node->isEmpty()) {
        // A normal node
        std::size_t NameLength = 0;
        if (std::optional<model::Function::Key> Key = Node->getFunction()) {
          auto Iterator = Binary.Functions().find(std::get<0>(*Key));
          revng_assert(Iterator != Binary.Functions().end());
          NameLength = Iterator->name().size();
        } else if (auto DynamicFunctionKey = Node->getDynamicFunction()) {
          const std::string &Key = std::get<0>(*DynamicFunctionKey);
          auto Iterator = Binary.ImportedDynamicFunctions().find(Key);
          revng_assert(Iterator != Binary.ImportedDynamicFunctions().end());
          NameLength = Iterator->name().size();
        } else {
          revng_abort("Unsupported node type.");
        }

        revng_assert(NameLength != 0);
        Node->Size = yield::layout::Size{
          NameLength * Configuration.LabelFontSize
            * Configuration.HorizontalFontFactor,
          1 * Configuration.LabelFontSize * Configuration.VerticalFontFactor
        };
      } else {
        // An entry node.
        Node->Size = yield::layout::Size{ 30, 30 };
      }

      Node->Size.W += Configuration.InternalNodeMarginSize * 2;
      Node->Size.H += Configuration.InternalNodeMarginSize * 2;
    }
  }

  std::string operator()(const yield::calls::PostLayoutNode &Node) const {
    if (Node.isEmpty())
      return "";

    std::string_view Location = Node.getLocationString();

    if (Node.IsShallow)
      return yield::ptml::shallowFunctionLink(ThePTMLBuilder, Location, Binary);

    if (!RootNodeLocation.has_value() || *RootNodeLocation == Location)
      return yield::ptml::functionNameDefinition(ThePTMLBuilder,
                                                 Location,
                                                 Binary);
    else
      return yield::ptml::functionLink(ThePTMLBuilder, Location, Binary);
  }
};

using CrossRelations = yield::crossrelations::CrossRelations;
std::string yield::svg::callGraph(const PTMLBuilder &ThePTMLBuilder,
                                  const CrossRelations &Relations,
                                  const model::Binary &Binary) {
  // TODO: make configuration accessible from outside.
  auto Configuration = cfg::Configuration::getDefault();
  Configuration.UseOrthogonalBends = false;
  constexpr auto LeftToRight = layout::sugiyama::Orientation::LeftToRight;
  constexpr auto BFS = layout::sugiyama::RankingStrategy::BreadthFirstSearch;

  LabelNodeHelper Helper{ ThePTMLBuilder, Binary, Configuration };

  yield::calls::PreLayoutGraph Result = Relations.toYieldGraph();
  auto EntryPoints = entryPoints(&Result);
  revng_assert(!EntryPoints.empty());
  if (EntryPoints.size() > 1) {
    // Add an artificial "root" node to make sure there's a single entry point.
    yield::calls::PreLayoutNode *Root = Result.addNode();
    for (yield::calls::PreLayoutNode *Entry : EntryPoints)
      Root->addSuccessor(Entry);
    Result.setEntryNode(Root);
  } else {
    Result.setEntryNode(EntryPoints.front());
  }

  auto Tree = calls::makeCalleeTree(Result);
  Helper.computeSizes(Tree);

  namespace sugiyama = layout::sugiyama;
  auto LT = sugiyama::compute(Tree, Configuration, LeftToRight, BFS, true);
  revng_assert(LT.has_value());

  return exportGraph<false>(ThePTMLBuilder,
                            *LT,
                            Configuration,
                            LeftToRight,
                            Helper);
}

static auto flipPoint(yield::layout::Point const &Point) {
  return yield::layout::Point{ -Point.X, -Point.Y };
};
static auto calculateDelta(yield::layout::Point const &LHS,
                           yield::layout::Point const &RHS) {
  return yield::layout::Point{ RHS.X - LHS.X, RHS.Y - LHS.Y };
}
static auto translatePoint(yield::layout::Point const &Point,
                           yield::layout::Point const &Delta) {
  return yield::layout::Point{ Point.X + Delta.X, Point.Y + Delta.Y };
}
static auto convertPoint(yield::layout::Point const &Point,
                         yield::layout::Point const &Delta) {
  return translatePoint(flipPoint(Point), Delta);
}

static yield::calls::PostLayoutGraph
combineHalvesHelper(std::string_view SlicePoint,
                    yield::calls::PostLayoutGraph &&ForwardsSlice,
                    yield::calls::PostLayoutGraph &&BackwardsSlice) {
  revng_assert(ForwardsSlice.size() != 0 && BackwardsSlice.size() != 0);

  auto IsSlicePoint = [&SlicePoint](const auto *Node) {
    return Node->getLocationString() == SlicePoint;
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
  using PostNode = yield::calls::PostLayoutGraph::Node;
  llvm::DenseMap<PostNode *, PostNode *> Lookup;
  auto AccessLookup = [&Lookup](PostNode *Key) {
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

std::string yield::svg::callGraphSlice(const PTMLBuilder &ThePTMLBuilder,
                                       std::string_view SlicePoint,
                                       const CrossRelations &Relations,
                                       const model::Binary &Binary) {
  // TODO: make configuration accessible from outside.
  auto Configuration = cfg::Configuration::getDefault();
  Configuration.UseOrthogonalBends = false;
  constexpr auto LeftToRight = layout::sugiyama::Orientation::LeftToRight;
  constexpr auto BFS = layout::sugiyama::RankingStrategy::BreadthFirstSearch;

  LabelNodeHelper Helper{ ThePTMLBuilder, Binary, Configuration, SlicePoint };

  // Ready the forwards facing part of the slice
  auto Forward = calls::makeCalleeTree(Relations.toYieldGraph(), SlicePoint);
  for (auto *From : Forward.nodes())
    for (auto [To, Label] : From->successor_edges())
      Label->IsBackwards = false;
  Helper.computeSizes(Forward);
  auto LaidOutForwardsGraph = layout::sugiyama::compute(Forward,
                                                        Configuration,
                                                        LeftToRight,
                                                        BFS,
                                                        true);
  revng_assert(LaidOutForwardsGraph.has_value());

  // Ready the backwards facing part of the slice
  auto Backwards = calls::makeCallerTree(Relations.toYieldGraph(), SlicePoint);
  for (auto *From : Backwards.nodes())
    for (auto [To, Label] : From->successor_edges())
      Label->IsBackwards = true;
  Helper.computeSizes(Backwards);
  auto LaidOutBackwardsGraph = layout::sugiyama::compute(Backwards,
                                                         Configuration,
                                                         LeftToRight,
                                                         BFS,
                                                         true);
  revng_assert(LaidOutBackwardsGraph.has_value());

  // Consume the halves to produce a combined graph and export it.
  auto CombinedGraph = combineHalvesHelper(SlicePoint,
                                           std::move(*LaidOutForwardsGraph),
                                           std::move(*LaidOutBackwardsGraph));
  return exportGraph<false>(ThePTMLBuilder,
                            CombinedGraph,
                            Configuration,
                            LeftToRight,
                            Helper);
}
