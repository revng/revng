#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Support/DOTGraphTraits.h"

#include "revng/ADT/GenericGraph.h"
#include "revng/Support/MetaAddress.h"

namespace efa {

struct BasicBlockNodeData {
  BasicBlockNodeData(MetaAddress Address) : Address(Address) {}
  MetaAddress Address;
};
using BasicBlockNode = BidirectionalNode<BasicBlockNodeData>;
using CallGraph = GenericGraph<BasicBlockNode>;

} // namespace efa

template<>
struct llvm::DOTGraphTraits<efa::CallGraph *>
  : public llvm::DefaultDOTGraphTraits {
  using EdgeIterator = llvm::GraphTraits<efa::CallGraph *>::ChildIteratorType;
  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getNodeLabel(const efa::BasicBlockNode *Node,
                                  const efa::CallGraph *Graph) {
    return Node->Address.toString();
  }

  static std::string getEdgeAttributes(const efa::BasicBlockNode *Node,
                                       const EdgeIterator EI,
                                       const efa::CallGraph *Graph) {
    return "color=black,style=dashed";
  }
};
