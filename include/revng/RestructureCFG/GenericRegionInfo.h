#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/CFG.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Support/Assert.h"
#include "revng/Support/GraphAlgorithms.h"

#include "GenericRegion.h"

template<class GraphT, class GT = llvm::GraphTraits<GraphT>>
class GenericRegionInfo {
public:
  using NodeT = typename GT::NodeRef;
  using Region = GenericRegion<NodeT>;

public:
  using region_container = llvm::SmallVector<
    std::unique_ptr<GenericRegion<NodeT>>>;
  using region_iterator = region_container::iterator;
  using region_const_iterator = region_container::const_iterator;
  using region_range = llvm::iterator_range<region_iterator>;
  using region_const_range = llvm::iterator_range<region_const_iterator>;

private:
  region_container Regions;

public:
  GenericRegionInfo() {}

  void compute(GraphT F);

  region_const_range regions() const {
    return llvm::make_range(Regions.begin(), Regions.end());
  }

  auto top_level_regions() const {

    // The top level regions are identified by the fact of having a `nullptr`
    // parent
    return make_filter_range(revng::dereferenceRange(regions()),
                             [](const Region &R) { return R.isRoot(); });
  }

  void clear() { Regions.clear(); }

private:
  void initializeRegions(GraphT F);
  void electHead(GraphT F);
  std::string print() const;
};

// Provide graph traits for the `Region` class
namespace llvm {

template<class NodeT>
struct GraphTraits<GenericRegion<NodeT> *> {
public:
  using Region = GenericRegionInfo<NodeT>::Region;
  using NodeRef = Region *;

public:
  static auto child_begin(NodeRef R) { return R->child_begin(); }

  static auto child_end(NodeRef R) { return R->child_end(); }

  static NodeRef getEntryNode(NodeRef R) { return R; }

public:
  using ChildIteratorType = decltype(child_begin(NodeRef()));
};

} // namespace llvm
