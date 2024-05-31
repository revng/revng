#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <cstddef>
#include <map>
#include <tuple>

#include "revng/Support/GraphAlgorithms.h"

#include "revng-c/RestructureCFG/CycleEquivalenceClass.h"

template<class NodeT>
class CycleEquivalenceResult {
private:
  static constexpr size_t SizeTMaxValue = std::numeric_limits<size_t>::max();

public:
  // They key of the result map is composed by the source/target node pair, and
  // the `SuccNum`, i.e., the index, in the successors vector of the source
  // node, of the edge
  using EdgeDescriptor = CycleEquivalenceClass<NodeT>::EdgeDescriptor;

private:
  using InternalMapType = std::map<EdgeDescriptor, size_t>;
  using map_iterator = InternalMapType::iterator;
  using map_const_iterator = InternalMapType::const_iterator;

public:
  /// The `NakedEdgeDescriptor` alias is used to refer to an `EdgeDescriptor`
  /// wrapping the original `NodeT` nodes
  using NakedEdgeDescriptor = revng::detail::EdgeDescriptor<NodeT>;
  using map_range = llvm::iterator_range<map_iterator>;
  using map_const_range = llvm::iterator_range<map_const_iterator>;

private:
  InternalMapType EdgeToCycleEquivalenceClassIDMap;

public:
  CycleEquivalenceResult() {}

  void insert(EdgeDescriptor Edge, size_t CycleEquivalenceClass) {
    revng_assert(not EdgeToCycleEquivalenceClassIDMap.contains(Edge));
    EdgeToCycleEquivalenceClassIDMap[Edge] = CycleEquivalenceClass;
  }

  /// Method to retrieve the `Cycle Equivalence Class ID` by passing as
  /// parameter the complete EdgeDescriptor (with the `SuccNum`)
  size_t getCycleEquivalenceClassID(EdgeDescriptor Edge) const {
    return EdgeToCycleEquivalenceClassIDMap.at(Edge);
  }

  /// Method to retrieve the `Cycle Equivalence Class ID" by passing as
  /// parameter the naked EdgeDescriptor (no `SuccNum`).
  /// Returns a range (that can be empty), containing the mapping for all the
  /// edges between the same nodepair described by the naked `Edge`.
  map_const_range getCycleEquivalenceClassID(NakedEdgeDescriptor Edge) const {

    // We return the range composed by the lower and upper bound of the map,
    // querying it with, as `SuccNum`, 0 and `SizeTMax`.
    // In this way, the return range will be:
    // 1) Empty if there is no entry for `Edge`.
    // 2) A range composed by a single element, if there is a single edge
    //    between the node pair.
    // 3) A range composed by multiple elements, if there are multiple edges
    //    between the same node pair.
    auto LowerBound = std::tuple_cat(Edge, std::make_tuple(0));
    auto UpperBound = std::tuple_cat(Edge, std::make_tuple(SizeTMaxValue));
    return llvm::make_range(EdgeToCycleEquivalenceClassIDMap
                              .lower_bound(LowerBound),
                            EdgeToCycleEquivalenceClassIDMap
                              .upper_bound(UpperBound));
  }

  /// This method, assumes that in the overall graph there are no multiple
  /// edges between the same node pair (and it asserts this property), and
  /// returns the exact equivalence class ID
  size_t getExactCycleEquivalenceClassID(NakedEdgeDescriptor Edge) const {
    map_const_range Range = getCycleEquivalenceClassID(Edge);
    size_t RangeSize = std::distance(Range.begin(), Range.end());
    revng_assert(RangeSize == 1);
    return Range.begin()->second;
  }
};
