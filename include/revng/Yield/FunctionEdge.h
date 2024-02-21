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
  using yield::generated::FunctionEdge::FunctionEdge;

  FunctionEdge() : yield::generated::FunctionEdge() { Type() = AssociatedType; }
  FunctionEdge(BasicBlockID Destination, FunctionEdgeType::Values Type) :
    yield::generated::FunctionEdge() {
    this->Destination() = Destination;
    this->Type() = Type;
  }

  explicit FunctionEdge(const efa::FunctionEdge &Source);

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/Yield/Generated/Late/FunctionEdge.h"
