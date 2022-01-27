#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/MutableSet.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Model/BasicBlock.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/FunctionType.h"
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
    type: std::string
    optional: true
  - name: Type
    doc: Type of the function
    type: model::FunctionType::Values
  - name: CFG
    doc: List of basic blocks, which represent the CFG
    sequence:
      type: SortedVector
      elementType: model::BasicBlock
    optional: true
  - name: StackFrameType
    doc: The type of the stack frame
    reference:
      pointeeType: model::Type
      rootType: model::Binary
    optional: true
  - name: Prototype
    doc: The prototype of the function
    reference:
      pointeeType: model::Type
      rootType: model::Binary
    optional: true
  - name: Attributes
    doc: Attributes for this call site
    sequence:
      type: MutableSet
      elementType: model::FunctionAttribute::Values
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

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/Model/Generated/Late/Function.h"
