#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/VerifyHelper.h"
#include "revng/Yield/FunctionEdgeBase.h"

/* TUPLE-TREE-YAML
name: FunctionEdge
doc: An edge on the CFG
type: struct
inherits: FunctionEdgeBase
fields: []
TUPLE-TREE-YAML */

namespace efa {
class FunctionEdge;
}

#include "revng/Yield/Generated/Early/FunctionEdge.h"

class yield::FunctionEdge : public yield::generated::FunctionEdge {
private:
  static constexpr const FunctionEdgeType::Values
    AssociatedType = FunctionEdgeType::DirectBranch;

public:
  FunctionEdge() : yield::generated::FunctionEdge() { Type = AssociatedType; }
  FunctionEdge(MetaAddress Destination, FunctionEdgeType::Values Type) :
    yield::generated::FunctionEdge(Destination, Type) {}

  explicit FunctionEdge(const efa::FunctionEdge &Source);

public:
  static bool classof(const FunctionEdgeBase *A) { return classof(A->key()); }
  static bool classof(const Key &K) {
    return not FunctionEdgeType::isCall(std::get<1>(K));
  }

  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/Yield/Generated/Late/FunctionEdge.h"
