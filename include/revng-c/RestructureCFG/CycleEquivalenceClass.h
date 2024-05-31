#pragma once

//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include <tuple>

#include "llvm/ADT/SmallSet.h"

#include "revng/Support/Assert.h"
#include "revng/Support/GraphAlgorithms.h"

#include "revng-c/ADT/STLExtras.h"

template<class NodeT>
class CycleEquivalenceClass {
public:
  using EdgeDescriptor = tuple_cat_t<revng::detail::EdgeDescriptor<NodeT>,
                                     make_tuple_t<size_t>>;

  using edge_container = llvm::SmallSet<EdgeDescriptor, 4>;
  using edge_const_iterator = edge_container::const_iterator;

private:
  size_t ID;
  edge_container Edges;

public:
  CycleEquivalenceClass(size_t ID) : ID(ID) {}

  size_t getID() const { return ID; };

  void insert(EdgeDescriptor &E) {
    revng_assert(not Edges.contains(E));
    Edges.insert(E);
  };

  std::string print() const {
    std::string Output;
    Output += "Bracket Equivalence Class ID: " + std::to_string(ID) + "\n";

    for (auto &Edge : Edges) {
      Output += std::get<0>(Edge)->getName().str() + ","
                + std::to_string(std::get<2>(Edge)) + " -> "
                + std::get<1>(Edge)->getName().str() + "\n";
    }

    return Output;
  };

  edge_const_iterator begin() const { return Edges.begin(); };
  edge_const_iterator end() const { return Edges.end(); };
};
