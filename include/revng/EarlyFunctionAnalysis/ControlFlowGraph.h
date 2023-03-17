#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>
#include <map>

#include "revng/ADT/Concepts.h"
#include "revng/ADT/GenericGraph.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadata.h"
#include "revng/Model/FunctionAttribute.h"

namespace efa {

namespace detail {
using SuccessorContainer = SortedVector<UpcastablePointer<FunctionEdgeBase>>;
}

template<typename T>
concept SpecializationOfBasicBlock = requires(T Instance) {
  { Instance.ID() } -> convertible_to<BasicBlockID>;
  { Instance.End() } -> convertible_to<MetaAddress>;
};

struct ParsedSuccessor {
  BasicBlockID NextInstructionAddress;
  MetaAddress OptionalCallAddress;
};

template<typename T>
inline ParsedSuccessor parseSuccessor(const T &Edge,
                                      const BasicBlockID &FallthroughAddress,
                                      const model::Binary &Binary) {
  using FunctionEdgeType = std::decay_t<decltype(Edge.Type())>;
  switch (Edge.Type()) {
  case FunctionEdgeType::DirectBranch:
  case FunctionEdgeType::Return:
  case FunctionEdgeType::BrokenReturn:
  case FunctionEdgeType::LongJmp:
  case FunctionEdgeType::Unreachable:
    return ParsedSuccessor{ .NextInstructionAddress = Edge.Destination(),
                            .OptionalCallAddress = MetaAddress::invalid() };

  case FunctionEdgeType::FunctionCall: {
    // Note: we assume that the first concrete type is the CallEdge. All of this
    //       hacks are necessary to handle identical data structures under
    //       different namespaces.
    using CallEdge = std::tuple_element_t<0, concrete_types_traits_t<T>>;
    auto *CE = llvm::cast<CallEdge>(&Edge);

    auto NextInstructionAddress = BasicBlockID::invalid();
    if (not CE->hasAttribute(Binary, model::FunctionAttribute::NoReturn)
        and not CE->IsTailCall()) {
      NextInstructionAddress = FallthroughAddress;
    }
    return ParsedSuccessor{ .NextInstructionAddress = NextInstructionAddress,
                            .OptionalCallAddress = Edge.Destination().start() };
  }
  case FunctionEdgeType::Killer:
    return ParsedSuccessor{ .NextInstructionAddress = BasicBlockID::invalid(),
                            .OptionalCallAddress = MetaAddress::invalid() };

  default:
  case FunctionEdgeType::Invalid:
  case FunctionEdgeType::Count:
    revng_abort();
    break;
  }
}

// clang-format off

/// A function for converting EFA's internal CFG representation into a generic
/// graph.
///
/// \param BB An arbitrary container of basic blocks that are verified
///        using the `IsBasicBlock` concept. These blocks are required to have
///        a start and an end addresses as well as a list of their successors.
///        It's expected for the graph it represents to be self contained, as
///        in "no block can ever reference another block that is not listed in
///        this container".
///
/// \param EntryAddress The `Start` address of the first block in the graph.
///
/// \param Binary The model of the binary, current function is a part of.
///        It's used for accessing the full function list (for the purpose of
///        identifying calls) as well as their attributes (like `noreturn`).
///
/// \tparam GraphType The type of the graph this function will make.
///         The node must be constructible from the type of basic block in
///         the \ref BB container as well as the entry address of the function.
///
/// \return A pair of the generic graph object (type of which is specified by
///         the first template parameter) and a map of all the basic block start
///         addresses to corresponding nodes that were created for them.
template<SpecializationOfGenericGraph GraphType,
         SpecializationOfBasicBlock BasicBlockType,
         typename... OtherTs,
         template<typename...>
         typename Container>
  requires std::is_constructible_v<typename GraphType::Node,
                                   const BasicBlockType &,
                                   const MetaAddress &>
std::pair<GraphType, std::map<BasicBlockID, typename GraphType::Node *>>
buildControlFlowGraph(const Container<BasicBlockType, OtherTs...> &BB,
                      const MetaAddress &EntryAddress,
                      const model::Binary &Binary) {
  // clang-format on
  using Node = typename GraphType::Node;
  std::pair<GraphType, std::map<BasicBlockID, Node *>> Res;

  auto &[Graph, NodeLookup] = Res;
  for (const BasicBlockType &Block : BB) {
    revng_assert(Block.ID().isValid());
    auto *NewNode = Graph.addNode(Node{ Block.ID(), EntryAddress });
    auto [_, Success] = NodeLookup.try_emplace(Block.ID(), NewNode);
    revng_assert(Success != false,
                 "Different basic blocks with the same `Start` address");
  }

  Node *ExitNode = nullptr;
  for (const BasicBlockType &Block : BB) {
    auto FromNodeIterator = NodeLookup.find(Block.ID());
    revng_assert(FromNodeIterator != NodeLookup.end());

    for (const auto &Edge : Block.Successors()) {
      auto [NextInstruction,
            _] = parseSuccessor(*Edge, Block.nextBlock(), Binary);
      if (NextInstruction.isValid()) {
        auto ToNodeIterator = NodeLookup.find(NextInstruction);
        revng_assert(ToNodeIterator != NodeLookup.end());
        FromNodeIterator->second->addSuccessor(ToNodeIterator->second);
      } else {
        if (ExitNode == nullptr) {
          constexpr auto Invalid = BasicBlockID::invalid();
          ExitNode = Graph.addNode(Node{ Invalid, EntryAddress });
          auto [_, Success] = NodeLookup.try_emplace(BasicBlockID::invalid(),
                                                     ExitNode);
          revng_assert(Success != false);
        }
        FromNodeIterator->second->addSuccessor(ExitNode);
      }
    }
  }

  revng_assert(EntryAddress.isValid());
  auto EntryNodeIterator = NodeLookup.find(BasicBlockID(EntryAddress));
  revng_assert(EntryNodeIterator != NodeLookup.end());
  Graph.setEntryNode(EntryNodeIterator->second);

  return Res;
}

} // namespace efa
