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
};
using Path = llvm::SmallVector<Point, 8>;

struct Size {
  Dimension W;
  Dimension H;
};

template<typename GraphType>
concept HasLLVMGraphTraits = requires(const GraphType &Graph) {
  {
    llvm::GraphTraits<GraphType>::getEntryNode(Graph)
  } -> std::convertible_to<typename llvm::GraphTraits<GraphType>::NodeRef>;
};

template<HasLLVMGraphTraits GraphType>
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

namespace detail {

template<typename GraphType>
using NR = typename llvm::GraphTraits<GraphType>::NodeRef;

template<typename GraphType>
using ER = typename llvm::GraphTraits<GraphType>::EdgeRef;

} // namespace detail

template<typename GraphType>
concept HasLayoutableInputGraphTraits = requires(detail::NR<GraphType> Node) {
  {
    LayoutableGraphTraits<GraphType>::getNodeSize(Node)
  } -> std::convertible_to<Size>;
} && HasLLVMGraphTraits<GraphType>;

template<typename GraphType>
concept HasLayoutableOutputGraphTraits = requires(detail::NR<GraphType> Node,
                                                  detail::ER<GraphType> Edge) {
  { LayoutableGraphTraits<GraphType>::setNodePosition(Node, Point()) };
  { LayoutableGraphTraits<GraphType>::setEdgePath(Edge, Path()) };
} && HasLLVMGraphTraits<GraphType>;

template<typename GraphType>
concept HasLayoutableGraphTraits = HasLLVMGraphTraits<GraphType>
                                   && HasLayoutableInputGraphTraits<GraphType>
                                   && HasLayoutableOutputGraphTraits<GraphType>;

} // namespace yield::layout
