#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/GraphLayout/Graphs.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/BasicBlockID.h"

namespace yield::calls {

class Node {
private:
  /// Stores a serialized `pipeline::Location` indicating what this node
  /// represents contextually. As of now, it's only allowed to be set to either
  /// `revng::ranks::Function` or `revng::ranks::DynamicFunction`, but that's
  /// likely to be extended in the future.
  std::string Location = {};

  using FunctionRank = decltype(revng::ranks::Function);
  using DynamicFunctionRank = decltype(revng::ranks::DynamicFunction);

public:
  bool IsShallow = false;

public:
  Node() = default;
  explicit Node(const pipeline::Location<FunctionRank> &Location) :
    Location(Location.toString()), IsShallow(false) {}
  Node &operator=(const pipeline::Location<FunctionRank> &NewLocation) {
    Location = NewLocation.toString();
    IsShallow = false;
    return *this;
  }

public:
  explicit Node(const pipeline::Location<DynamicFunctionRank> &Location) :
    Location(Location.toString()), IsShallow(false) {}
  Node &operator=(const pipeline::Location<DynamicFunctionRank> &NewLocation) {
    Location = NewLocation.toString();
    IsShallow = false;
    return *this;
  }

public:
  bool isEmpty() const { return Location.empty(); }

  std::optional<FunctionRank::Type> getFunction() const {
    auto Result = pipeline::locationFromString(revng::ranks::Function,
                                               Location);
    if (!Result.has_value())
      return std::nullopt;

    return Result->at(revng::ranks::Function);
  }

  std::optional<DynamicFunctionRank::Type> getDynamicFunction() const {
    auto Result = pipeline::locationFromString(revng::ranks::DynamicFunction,
                                               Location);
    if (!Result.has_value())
      return std::nullopt;

    return Result->at(revng::ranks::DynamicFunction);
  }

  const std::string &getLocationString() const { return Location; }
};

struct Edge {
  bool IsBackwards = false;
};

using PreLayoutGraph = layout::InputGraph<Node, Edge>;
using PreLayoutNode = PreLayoutGraph::Node;

using PostLayoutGraph = layout::OutputGraph<Node, Edge>;
using PostLayoutNode = PostLayoutGraph::Node;

} // namespace yield::calls
