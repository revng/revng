#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/CommonFunctionMethods.h"
#include "revng/Model/TypeDefinition.h"
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
    type: Type
    upcastable: true
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

class model::CallSitePrototype
  : public model::generated::CallSitePrototype,
    public model::CommonFunctionMethods<CallSitePrototype> {
public:
  using generated::CallSitePrototype::CallSitePrototype;
  CallSitePrototype(MetaAddress CallerAddress,
                    model::UpcastableType &&Prototype,
                    bool IsTailCall) :
    generated::CallSitePrototype(CallerAddress,
                                 std::move(Prototype),
                                 IsTailCall,
                                 {}) {}

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  bool isDirect() const { return Prototype().empty(); }
};

#include "revng/Model/Generated/Late/CallSitePrototype.h"
