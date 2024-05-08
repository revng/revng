#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/ADT/PostOrderIterator.h"

template<class GraphT,
         class GT = llvm::GraphTraits<GraphT>,
         class SetType = std::set<typename llvm::GraphTraits<GraphT>::NodeRef>>
class ReversePostOrderTraversalExt {
  using NodeRef = typename GT::NodeRef;
  using NodeVec = std::vector<NodeRef>;

  NodeVec Blocks; // Block list in normal RPO order

  void initialize(GraphT G, SetType &WhiteList) {
    std::copy(po_ext_begin(G, WhiteList),
              po_ext_end(G, WhiteList),
              std::back_inserter(Blocks));
  }

public:
  using rpo_iterator = typename NodeVec::reverse_iterator;
  using const_rpo_iterator = typename NodeVec::const_reverse_iterator;

  ReversePostOrderTraversalExt(GraphT G, SetType &WhiteList) {
    initialize(G, WhiteList);
  }

  // Because we want a reverse post order, use reverse iterators from the vector
  rpo_iterator begin() { return Blocks.rbegin(); }
  const_rpo_iterator begin() const { return Blocks.crbegin(); }
  rpo_iterator end() { return Blocks.rend(); }
  const_rpo_iterator end() const { return Blocks.crend(); }
};
