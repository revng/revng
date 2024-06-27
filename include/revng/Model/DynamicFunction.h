#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/MutableSet.h"
#include "revng/Model/CommonFunctionMethods.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: DynamicFunction
doc: Function defined in a dynamic library
type: struct
fields:
  - name: CustomName
    doc: An optional custom name
    type: Identifier
    optional: true
  - name: OriginalName
    doc: The name of the symbol for this dynamic function
    type: string
  - name: Comment
    type: string
    optional: true
  - name: Prototype
    doc: The prototype of the function
    type: Type
    upcastable: true
    optional: true
  - name: Attributes
    doc: Function attributes
    sequence:
      type: MutableSet
      elementType: FunctionAttribute
    optional: true
  - name: Relocations
    sequence:
      type: SortedVector
      elementType: Relocation
    optional: true

key:
  - OriginalName
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/DynamicFunction.h"

class model::DynamicFunction
  : public model::generated::DynamicFunction,
    public model::CommonFunctionMethods<DynamicFunction> {
public:
  using generated::DynamicFunction::DynamicFunction;

public:
  Identifier name() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/DynamicFunction.h"
