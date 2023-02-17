#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/MutableSet.h"
#include "revng/EarlyFunctionAnalysis/FunctionEdgeBase.h"
#include "revng/Model/Binary.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/VerifyHelper.h"

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

#include "revng/EarlyFunctionAnalysis/Generated/Early/CallEdge.h"

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
  static bool classof(const FunctionEdgeBase *A) { return classof(A->key()); }
  static bool classof(const Key &K) {
    return std::get<0>(K) == FunctionEdgeBaseKind::CallEdge;
  }

public:
  bool hasAttribute(const model::Binary &Binary,
                    model::FunctionAttribute::Values Attribute) const {
    using namespace model;

    if (Attributes().count(Attribute) != 0)
      return true;

    if (const auto *CalleeAttributes = calleeAttributes(Binary))
      return CalleeAttributes->count(Attribute) != 0;
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
  const MutableSet<model::FunctionAttribute::Values> *
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
