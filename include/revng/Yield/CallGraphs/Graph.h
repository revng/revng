#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/GraphLayout/Graphs.h"
#include "revng/Support/BasicBlockID.h"

namespace yield::calls {

struct Node {
  BasicBlockID Address;
  BasicBlockID NextAddress;

  explicit Node(BasicBlockID Address = BasicBlockID::invalid()) :
    Address(std::move(Address)) {}

  bool isEmpty() const { return !Address.isValid(); }
};

struct Edge {
  bool IsBackwards = false;
};

using PreLayoutGraph = layout::InputGraph<Node, Edge>;
using PreLayoutNode = PreLayoutGraph::Node;

using PostLayoutGraph = layout::OutputGraph<Node, Edge>;
using PostLayoutNode = PostLayoutGraph::Node;

} // namespace yield::calls
