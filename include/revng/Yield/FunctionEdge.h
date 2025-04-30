#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Yield/FunctionEdgeBase.h"

namespace efa {
class FunctionEdge;
}

#include "revng/Yield/Generated/Early/FunctionEdge.h"

namespace model {
class VerifyHelper;
}

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
};

#include "revng/Yield/Generated/Late/FunctionEdge.h"
