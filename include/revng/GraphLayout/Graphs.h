#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <unordered_map>

#include "revng/ADT/GenericGraph.h"
#include "revng/GraphLayout/Traits.h"

namespace yield::layout {

namespace detail {

template<typename DataType>
struct InputNode : public DataType {
  Size Size = { 0, 0 };

  using DataType::DataType;
  InputNode(const DataType &Another) : DataType(Another){};
  InputNode(DataType &&Another) : DataType(Another){};
};

template<typename DataType>
struct OutputNode : public DataType {
  Point Center = { 0, 0 };
  Size Size = { 0, 0 };

  using DataType::DataType;
  OutputNode(const DataType &Another) : DataType(Another){};
  OutputNode(DataType &&Another) : DataType(Another){};
};

template<typename DataType>
struct InputEdge : public DataType {
  using DataType::DataType;
  InputEdge(const DataType &Another) : DataType(Another){};
  InputEdge(DataType &&Another) : DataType(Another){};
};

template<typename DataType>
struct OutputEdge : public DataType {
  Path Path = {};

  using DataType::DataType;
  OutputEdge(const DataType &Another) : DataType(Another){};
  OutputEdge(DataType &&Another) : DataType(Another){};
};

} // namespace detail

template<typename NodeDataType, typename EdgeDataType = Empty>
using InputNode = MutableEdgeNode<detail::InputNode<NodeDataType>,
                                  detail::InputEdge<EdgeDataType>,
                                  false>;

template<typename NodeDataType, typename EdgeDataType = Empty>
class InputGraph
  : public GenericGraph<InputNode<NodeDataType, EdgeDataType>, 16, true> {
  using Base = GenericGraph<InputNode<NodeDataType, EdgeDataType>, 16, true>;

public:
  using Base::Base;
};

template<typename NodeDataType, typename EdgeDataType>
using OutputNode = MutableEdgeNode<detail::OutputNode<NodeDataType>,
                                   detail::OutputEdge<EdgeDataType>,
                                   false>;

template<typename NodeDataType, typename EdgeDataType>
class OutputGraph
  : public GenericGraph<OutputNode<NodeDataType, EdgeDataType>, 16, true> {
  using Base = GenericGraph<OutputNode<NodeDataType, EdgeDataType>, 16, true>;

public:
  using Base::Base;
};

template<typename NodeDataType, typename EdgeDataType>
struct LayoutableGraphTraits<InputGraph<NodeDataType, EdgeDataType> *> {
  using GraphType = InputGraph<NodeDataType, EdgeDataType> *;
  static_assert(SpecializationOfGenericGraph<std::remove_pointer_t<GraphType>>);
  using LLVMTrait = llvm::GraphTraits<GraphType>;

  static const Size &getNodeSize(typename LLVMTrait::NodeRef Node) {
    return Node.Size;
  }
};

template<typename NodeDataType, typename EdgeDataType>
struct LayoutableGraphTraits<OutputGraph<NodeDataType, EdgeDataType> *> {
  using GraphType = OutputGraph<NodeDataType, EdgeDataType> *;
  static_assert(SpecializationOfGenericGraph<std::remove_pointer_t<GraphType>>);
  using LLVMTrait = llvm::GraphTraits<GraphType>;

  static const Size &getNodeSize(typename LLVMTrait::NodeRef Node) {
    return Node->Size;
  }

  static void setNodePosition(typename LLVMTrait::NodeRef Node, Point &&Point) {
    Node->Center = std::move(Point);
  }
  static void setEdgePath(typename LLVMTrait::EdgeRef Edge, Path &&Path) {
    Edge.Label->Path = std::move(Path);
  }
};

namespace detail {

template<typename NodeDataType, typename EdgeDataType>
OutputGraph<NodeDataType, EdgeDataType>
convert(const InputGraph<NodeDataType, EdgeDataType> &Input) {
  OutputGraph<NodeDataType, EdgeDataType> Result;

  using InputNode = layout::InputNode<NodeDataType, EdgeDataType>;
  using OutputNode = layout::OutputNode<NodeDataType, EdgeDataType>;

  std::unordered_map<const InputNode *, OutputNode *> Lookup;
  for (const InputNode *Node : Input.nodes()) {
    OutputNode Output(*Node);
    Output.Size = Node->Size;

    auto [It, Success] = Lookup.emplace(Node,
                                        Result.addNode(std::move(Output)));
    revng_assert(Success);
  }

  for (const InputNode *From : Input.nodes())
    for (auto [To, EdgeLabel] : From->successor_edges())
      Lookup.at(From)->addSuccessor(Lookup.at(To),
                                    EdgeDataType(std::move(*EdgeLabel)));

  return Result;
}

} // namespace detail

} // namespace yield::layout
