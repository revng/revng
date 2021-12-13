#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/CABIFunctionType.h"
#include "revng/Model/EnumType.h"
#include "revng/Model/FunctionEdgeType.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Model/StructType.h"
#include "revng/Model/TypedefType.h"
#include "revng/Model/UnionType.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"

/* TUPLE-TREE-YAML
name: FunctionEdgeBase
doc: An edge on the CFG
type: struct
fields:
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
    type: FunctionEdgeType::Values
key:
  - Destination
  - Type
abstract: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/FunctionEdgeBase.h"

class model::FunctionEdgeBase : public model::generated::FunctionEdgeBase {
public:
  using generated::FunctionEdgeBase::FunctionEdgeBase;

public:
  static bool classof(const FunctionEdgeBase *A) { return classof(A->key()); }
  static bool classof(const Key &K) { return true; }

  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/Model/Generated/Late/FunctionEdgeBase.h"
