#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/CFG.h"

#include "revng/Support/Assert.h"
#include "revng/Support/GraphAlgorithms.h"

#include "revng-c/ADT/STLExtras.h"

template<class NodeT>
class GenericRegion {
public:
  // It is very important that we use the `SetVector` version of the container
  // here. The first needed property is uniqueness, and the second is the
  // determinism given by the guarantee on the iteration order based on the
  // insertion order.
  using block_container = llvm::SmallSetVector<NodeT, 4>;
  using block_iterator = block_container::iterator;
  using block_const_iterator = block_container::const_iterator;
  using block_range = llvm::iterator_range<block_iterator>;
  using block_const_range = llvm::iterator_range<block_const_iterator>;

  // We use a `SetVector` variant of the container here too, in order to have
  // guarantess on uniqueness and iteration determinism based on the insertion
  // order
  using child_container = llvm::SmallSetVector<GenericRegion *, 4>;
  using child_iterator = child_container::iterator;
  using child_range = llvm::iterator_range<child_iterator>;

  using edge = tuple_cat_t<revng::detail::EdgeDescriptor<NodeT>,
                           make_tuple_t<size_t>>;
  using edge_container = llvm::SmallSetVector<NodeT, 4>;
  using edge_iterator = edge_container::iterator;
  using edge_range = llvm::iterator_range<edge_iterator>;

private:
  block_container Blocks;

  NodeT Head = nullptr;

  // The parent `GenericRegion`. Is null for every root `GenericRegion`.
  GenericRegion *ParentRegion = nullptr;

  child_container Children;

  edge_container RetreatingEdges;

public:
  GenericRegion() {}

  void insertBlock(NodeT Block) { Blocks.insert(Block); }

  bool containsBlock(NodeT Block) { return Blocks.contains(Block); }

  block_iterator block_begin() { return Blocks.begin(); }
  block_const_iterator block_begin() const { return Blocks.begin(); }

  block_iterator block_end() { return Blocks.end(); }
  block_const_iterator block_end() const { return Blocks.end(); }

  block_range blocks() { return llvm::make_range(block_begin(), block_end()); }
  block_const_range blocks() const {
    return llvm::make_range(block_begin(), block_end());
  }

  NodeT getHead() { return Head; }

  void setHead(NodeT Head) { this->Head = Head; }

  void setParent(GenericRegion *NewParent) {
    revng_assert(ParentRegion == nullptr);
    ParentRegion = NewParent;
  }

  GenericRegion getParent() const { return ParentRegion; }

  bool isRoot() const { return ParentRegion == nullptr; }

  /// Helper method to insert a child `GenericRegion` to this
  void addChild(GenericRegion *Child) {

    // Populate the `Children` vector with a naked pointer of the child.
    Children.insert(Child);

    // Insert in the `Child` the reference to the parent `GenericRegion`
    Child->setParent(this);
  }

  child_iterator child_begin() { return Children.begin(); }
  child_iterator child_end() { return Children.end(); }
  child_range children() {
    return llvm::make_range(child_begin(), child_end());
  }

  void addRetreating(edge Edge) { RetreatingEdges.insert(Edge); }

  edge_iterator retreating_begin() { return RetreatingEdges.begin(); }
  edge_iterator retreating_end() { return RetreatingEdges.end(); }
  edge_range retreatings() {
    return llvm::make_range(retreating_begin(), retreating_end());
  }
};
