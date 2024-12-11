#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/MutableSet.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Model/CallSitePrototype.h"
#include "revng/Model/CommonFunctionMethods.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

/* TUPLE-TREE-YAML
name: Function
doc: A function defined within this binary.
type: struct
fields:
  - name: Entry
    doc: |
      The address of the entry point.

      Note: this does not necessarily correspond to the address of the basic
      block with the lowest address.
    type: MetaAddress
  - name: CustomName
    type: Identifier
    optional: true
  - name: OriginalName
    type: string
    optional: true
  - name: Comment
    type: string
    optional: true
  - name: StackFrameType
    doc: The type of the stack frame.
    type: Type
    upcastable: true
    optional: true
  - name: Prototype
    doc: The prototype of the function.
    type: Type
    upcastable: true
    optional: true
  - name: Attributes
    doc: Attributes for this call site.
    sequence:
      type: MutableSet
      elementType: FunctionAttribute
    optional: true
  - name: CallSitePrototypes
    doc: |-
      Information about specific call sites within this function.
    sequence:
      type: SortedVector
      elementType: CallSitePrototype
    optional: true
  - name: ExportedNames
    doc: |-
      The list of names used to make this function available as a dynamic
      symbol.
    sequence:
      type: SortedVector
      elementType: string
    optional: true
key:
  - Entry
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/Function.h"

class model::Function : public model::generated::Function,
                        public model::CommonFunctionMethods<Function> {
public:
  using generated::Function::Function;

public:
  /// The helper for stack frame type unwrapping.
  /// Use this when you need to access/modify the existing struct,
  /// and \ref StackFrameType() when you need to assign a new one.
  model::StructDefinition *stackFrameType() {
    if (StackFrameType().isEmpty())
      return nullptr;
    else
      return StackFrameType()->getStruct();
  }

  /// The helper for stack argument type unwrapping.
  /// Use this when you need to access/modify the existing struct,
  /// and \ref StackFrameType() when you need to assign a new one.
  const model::StructDefinition *stackFrameType() const {
    if (StackFrameType().isEmpty())
      return nullptr;
    else
      return StackFrameType()->getStruct();
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dumpTypeGraph(const char *Path,
                     const model::Binary &Binary) const debug_function;
};

#include "revng/Model/Generated/Late/Function.h"
