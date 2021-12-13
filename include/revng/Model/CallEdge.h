#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/MutableSet.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/FunctionEdgeBase.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: CallEdge
doc: A CFG edge to represent function calls (direct, indirect and tail calls)
type: struct
inherits: FunctionEdgeBase
fields:
  - name: Prototype
    doc: |
      Path to the prototype for this call site

      In case of a direct function call, it has to be invalid.
    reference:
      pointeeType: "model::Type"
      rootType: "model::Binary"
  - name: DynamicFunction
    doc: |
      Name of the dynamic function being called, or empty if not a dynamic call
    type: "std::string"
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

#include "revng/Model/Generated/Early/CallEdge.h"

class model::CallEdge : public model::generated::CallEdge {
public:
  using generated::CallEdge::CallEdge;

public:
  static bool classof(const FunctionEdgeBase *A) { return classof(A->key()); }
  static bool classof(const Key &K) {
    return FunctionEdgeType::isCall(std::get<1>(K));
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/Model/Generated/Late/CallEdge.h"
