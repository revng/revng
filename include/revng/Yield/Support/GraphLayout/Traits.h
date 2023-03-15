#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <type_traits>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/ADT/Concepts.h"

namespace yield::layout {

using Coordinate = float;
using Dimension = Coordinate;

struct Point {
  Coordinate X;
  Coordinate Y;

  constexpr Point(Coordinate X = 0, Coordinate Y = 0) : X(X), Y(Y) {}
};
using Path = llvm::SmallVector<Point, 8>;

struct Size {
  Dimension W;
  Dimension H;

  constexpr Size(Dimension W = 0, Dimension H = 0) : W(W), H(H) {}
};

// clang-format off
template<typename GraphType>
concept HasLLVMGraphTraits = requires(const GraphType &Graph) {
  { llvm::GraphTraits<GraphType>::getEntryNode(Graph) } ->
    convertible_to<typename llvm::GraphTraits<GraphType>::NodeRef>;
};

template <HasLLVMGraphTraits GraphType>
struct LayoutableGraphTraits {
  using LLVMTrait = llvm::GraphTraits<GraphType>;

  // Elements to provide to qualify as an input graph:
  //
  // const Size &getNodeSize(typename LLVMTrait::NodeRef Node);

  // Elements to provide to qualify as an output graph:
  //
  // void setNodePosition(typename LLVMTrait::NodeRef Node, Point &&Point);
  // void setEdgePath(typename LLVMTrait::EdgeRef Edge, Path &&Path);

  // Note that a single graph is allowed to provide everything, so it can
  // qualify as both (for computing the layout in-place).
};

template <typename GraphType>
concept HasLayoutableInputGraphTraits = HasLLVMGraphTraits<GraphType>
  && requires(typename llvm::GraphTraits<GraphType>::NodeRef Node) {
    { LayoutableGraphTraits<GraphType>::getNodeSize(Node) } ->
      convertible_to<Size>;
  };

template <typename GraphType>
concept HasLayoutableOutputGraphTraits = HasLLVMGraphTraits<GraphType>
  && requires(typename llvm::GraphTraits<GraphType>::NodeRef Node,
              typename llvm::GraphTraits<GraphType>::EdgeRef Edge) {
    { LayoutableGraphTraits<GraphType>::setNodePosition(Node, Point()) };
    { LayoutableGraphTraits<GraphType>::setEdgePath(Edge, Path()) };
  };

template<typename GraphType>
concept HasLayoutableGraphTraits = HasLLVMGraphTraits<GraphType>
                                   && HasLayoutableInputGraphTraits<GraphType>
                                   && HasLayoutableOutputGraphTraits<GraphType>;
// clang-format on

} // namespace yield::layout
