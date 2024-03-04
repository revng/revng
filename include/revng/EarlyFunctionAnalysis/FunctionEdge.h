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
  using generated::FunctionEdge::FunctionEdge;

  FunctionEdge() : efa::generated::FunctionEdge() { Type() = AssociatedType; }
  FunctionEdge(BasicBlockID Destination, FunctionEdgeType::Values Type) :
    efa::generated::FunctionEdge() {
    this->Destination() = Destination;
    this->Type() = Type;
  }

  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
};

#include "revng/EarlyFunctionAnalysis/Generated/Late/FunctionEdge.h"
