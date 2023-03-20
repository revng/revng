#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/GraphLayout/Graphs.h"
#include "revng/Support/BasicBlockID.h"

namespace yield::cfg {

struct Node {
  BasicBlockID Address;
  BasicBlockID NextAddress;

  explicit Node(BasicBlockID Address = BasicBlockID::invalid()) :
    Address(std::move(Address)) {}

  bool isEmpty() const { return !Address.isValid(); }
};

enum class EdgeType { Unconditional, Call, Taken, Refused };
struct Edge {
  EdgeType Type = EdgeType::Unconditional;
};

using PreLayoutNode = layout::InputNode<Node, Edge>;
using PreLayoutGraph = layout::InputGraph<Node, Edge>;

using PostLayoutNode = layout::OutputNode<Node, Edge>;
using PostLayoutGraph = layout::OutputGraph<Node, Edge>;

} // namespace yield::cfg
