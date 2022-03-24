#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/FunctionEdgeBase.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: FunctionEdge
doc: An edge on the CFG
type: struct
inherits: FunctionEdgeBase
fields: []
TUPLE-TREE-YAML */

#include "revng/EarlyFunctionAnalysis/Generated/Early/FunctionEdge.h"

class efa::FunctionEdge : public efa::generated::FunctionEdge {
private:
  static constexpr const FunctionEdgeType::Values
    AssociatedType = FunctionEdgeType::DirectBranch;

public:
  FunctionEdge() : efa::generated::FunctionEdge() { Type = AssociatedType; }
  FunctionEdge(MetaAddress Destination, FunctionEdgeType::Values Type) :
    efa::generated::FunctionEdge(Destination, Type) {}

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

#include "revng/EarlyFunctionAnalysis/Generated/Late/FunctionEdge.h"
