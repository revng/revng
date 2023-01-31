#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/FunctionEdgeType.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"

/* TUPLE-TREE-YAML
name: FunctionEdgeBase
doc: An edge on the CFG
type: struct
fields:
  - name: Kind
    type: FunctionEdgeBaseKind
  - name: Destination
    doc: |
      Target of the CFG edge

      If invalid, it's an indirect edge such as a return instruction or an
      indirect function call.
      If valid, it's either the address of the basic block in case of a direct
      branch, or, in case of a function call, the address of the callee.
      TODO: switch to TupleTreeReference
    type: MetaAddress
  - name: Type
    doc: Type of the CFG edge
    type: FunctionEdgeType
key:
  - Kind
  - Destination
abstract: true
TUPLE-TREE-YAML */

#include "revng/EarlyFunctionAnalysis/Generated/Early/FunctionEdgeBase.h"

class efa::FunctionEdgeBase : public efa::generated::FunctionEdgeBase {
public:
  using generated::FunctionEdgeBase::FunctionEdgeBase;

public:
  static bool classof(const FunctionEdgeBase *A) { return classof(A->key()); }
  static bool classof(const Key &K) { return true; }

public:
  bool isDirect() const { return Destination().isValid(); }
  bool isIndirect() const { return not isDirect(); }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/EarlyFunctionAnalysis/Generated/Late/FunctionEdgeBase.h"
