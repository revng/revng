#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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
    doc: Prototype
    reference:
      pointeeType: TypeDefinition
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

  /// Get the actual prototype, skipping any typedefs
  model::DefinitionReference prototype() const {
    return model::QualifiedType::getFunctionType(Prototype()).value();
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  bool isDirect() const { return Prototype().empty(); }
};

#include "revng/Model/Generated/Late/CallSitePrototype.h"
