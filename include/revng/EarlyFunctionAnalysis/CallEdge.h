#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/MutableSet.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdgeBase.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionAttribute.h"

#include "revng/EarlyFunctionAnalysis/Generated/Early/CallEdge.h"

namespace model {
class VerifyHelper;
}

class efa::CallEdge : public efa::generated::CallEdge {
private:
  static constexpr const FunctionEdgeType::Values
    AssociatedType = FunctionEdgeType::FunctionCall;

public:
  using generated::CallEdge::CallEdge;

  CallEdge() : efa::generated::CallEdge() { Type() = AssociatedType; }

  CallEdge(BasicBlockID Destination, FunctionEdgeType::Values Type) :
    efa::generated::CallEdge() {
    this->Destination() = Destination;
    this->Type() = Type;
  }

public:
  bool hasAttribute(const model::Binary &Binary,
                    model::FunctionAttribute::Values Attribute) const {
    using namespace model;

    if (Attributes().contains(Attribute))
      return true;

    if (const auto *CalleeAttributes = calleeAttributes(Binary))
      return CalleeAttributes->contains(Attribute);
    else
      return false;
  }

  MutableSet<model::FunctionAttribute::Values>
  attributes(const model::Binary &Binary) const {
    MutableSet<model::FunctionAttribute::Values> Result;
    auto Inserter = Result.batch_insert();
    for (auto &Attribute : Attributes())
      Inserter.insert(Attribute);

    if (const auto *CalleeAttributes = calleeAttributes(Binary))
      for (auto &Attribute : *CalleeAttributes)
        Inserter.insert(Attribute);

    return Result;
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;

private:
  const TrackingMutableSet<model::FunctionAttribute::Values> *
  calleeAttributes(const model::Binary &Binary) const {
    if (not DynamicFunction().empty()) {
      const auto &F = Binary.ImportedDynamicFunctions().at(DynamicFunction());
      return &F.Attributes();
    } else if (Destination().isValid()) {
      return &Binary.Functions().at(Destination().start()).Attributes();
    } else {
      return nullptr;
    }
  }
};

#include "revng/EarlyFunctionAnalysis/Generated/Late/CallEdge.h"
