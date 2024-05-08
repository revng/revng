#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/ADT/MutableSet.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Model/CallSitePrototype.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

/* TUPLE-TREE-YAML
name: Function
doc: A function
type: struct
fields:
  - name: Entry
    doc: |
      The address of the entry point

      \note This does not necessarily correspond to the address of the basic
      block with the lowest address.
    type: MetaAddress
  - name: CustomName
    doc: An optional custom name
    type: Identifier
    optional: true
  - name: OriginalName
    type: string
    optional: true
  - name: Comment
    type: string
    optional: true
  - name: StackFrameType
    doc: The type of the stack frame
    reference:
      pointeeType: Type
      rootType: Binary
    optional: true
  - name: Prototype
    doc: The prototype of the function
    reference:
      pointeeType: Type
      rootType: Binary
    optional: true
  - name: Attributes
    doc: Attributes for this call site
    sequence:
      type: MutableSet
      elementType: FunctionAttribute
    optional: true
  - name: CallSitePrototypes
    sequence:
      type: SortedVector
      elementType: CallSitePrototype
    optional: true
  - name: ExportedNames
    sequence:
      type: SortedVector
      elementType: string
    optional: true
key:
  - Entry
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Function.h"

class model::Function : public model::generated::Function {
public:
  using generated::Function::Function;

public:
  Identifier name() const;

  model::TypePath prototype(const model::Binary &Root) const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
  void dumpTypeGraph(const char *Path) const debug_function;
};

#include "revng/Model/Generated/Late/Function.h"
