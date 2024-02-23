#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/MutableSet.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Yield/FunctionEdge.h"

/* TUPLE-TREE-YAML
name: CallEdge
doc: A CFG edge to represent function calls (direct, indirect and tail calls)
type: struct
inherits: FunctionEdgeBase
fields:
  - name: DynamicFunction
    doc: |
      Name of the dynamic function being called, or empty if not a dynamic call
    type: string
    optional: true
  - name: IsTailCall
    doc: Is this a tail call?
    type: bool
    optional: true
  - name: Attributes
    doc: |
      Attributes for this function

      Note: To have the effective list of attributes for this call site, you
      have to add attributes on the called function.
      TODO: switch to std::set
    sequence:
      type: MutableSet
      elementType: model::FunctionAttribute::Values
    optional: true
TUPLE-TREE-YAML */

#include "revng/Yield/Generated/Early/CallEdge.h"

namespace efa {
class CallEdge;
}

class yield::CallEdge : public yield::generated::CallEdge {
private:
  static constexpr const FunctionEdgeType::Values
    AssociatedType = FunctionEdgeType::FunctionCall;

public:
  using yield::generated::CallEdge::CallEdge;

  CallEdge() : yield::generated::CallEdge() { Type() = AssociatedType; }

  CallEdge(BasicBlockID Destination, FunctionEdgeType::Values Type) :
    yield::generated::CallEdge() {
    this->Destination() = Destination;
    this->Type() = Type;
  }

  CallEdge(MetaAddress Destination, FunctionEdgeType::Values Type) :
    yield::generated::CallEdge() {
    this->Destination() = BasicBlockID(Destination);
    this->Type() = Type;
  }

  explicit CallEdge(const efa::CallEdge &Source);

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
  void dump() const debug_function;

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

#include "revng/Yield/Generated/Late/CallEdge.h"
