#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Type.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

/* TUPLE-TREE-YAML
name: CallSitePrototype
doc: Prototype of a callsite
type: struct
fields:
  - name: CallerBlockAddress
    doc: Address of the basic block of the call
    type: MetaAddress
  - name: Prototype
    doc: Prototype
    reference:
      pointeeType: Type
      rootType: Binary
  - name: IsTailCall
    doc: Whether this call site is a tail call or not
    type: bool
    optional: true
  - name: Attributes
    doc: Attributes for this call site
    sequence:
      type: MutableSet
      elementType: FunctionAttribute
    optional: true
key:
  - CallerBlockAddress
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/CallSitePrototype.h"

class model::CallSitePrototype : public model::generated::CallSitePrototype {
public:
  using generated::CallSitePrototype::CallSitePrototype;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
  bool isDirect() const { return not Prototype().isValid(); }
};

#include "revng/Model/Generated/Late/CallSitePrototype.h"
