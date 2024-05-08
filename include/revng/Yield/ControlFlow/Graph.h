#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/GraphLayout/Graphs.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/BasicBlockID.h"

namespace yield::cfg {

class Node {
private:
  /// Stores a serialized `pipeline::Location` indicating what this node
  /// represents contextually. As of now, it's only allowed to be set to
  /// `revng::ranks::BasicBlock` but that's likely to be extended in the future.
  std::string Location = {};

private:
  using BasicBlockRank = decltype(revng::ranks::BasicBlock);

public:
  Node() = default;
  Node(BasicBlockID Address,
       const MetaAddress &Function = MetaAddress::invalid()) :
    Location(pipeline::serializedLocation(revng::ranks::BasicBlock,
                                          Function,
                                          Address)) {}
  explicit Node(const pipeline::Location<BasicBlockRank> &Location) :
    Location(Location.toString()) {}
  Node &operator=(const pipeline::Location<BasicBlockRank> &NewLocation) {
    Location = NewLocation.toString();
    return *this;
  }

public:
  bool isEmpty() const { return Location.empty(); }

  BasicBlockRank::Type getBasicBlock() const {
    revng_assert(!isEmpty());

    auto MaybeResult = pipeline::locationFromString(revng::ranks::BasicBlock,
                                                    Location);
    revng_assert(MaybeResult.has_value());

    return MaybeResult->at(revng::ranks::BasicBlock);
  }
};

enum class EdgeType {
  Unconditional,
  Call,
  Taken,
  Refused
};
struct Edge {
  EdgeType Type = EdgeType::Unconditional;
};

using PreLayoutNode = layout::InputNode<Node, Edge>;
using PreLayoutGraph = layout::InputGraph<Node, Edge>;

using PostLayoutNode = layout::OutputNode<Node, Edge>;
using PostLayoutGraph = layout::OutputGraph<Node, Edge>;

} // namespace yield::cfg
