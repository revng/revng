#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: StructField
doc: A field of a struct type in model, with offset, qualified type, and name
type: struct
fields:
  - name: Offset
    type: uint64_t
  - name: CustomName
    type: Identifier
    optional: true
  - name: OriginalName
    type: string
    optional: true
  - name: Comment
    type: string
    optional: true
  - name: Type
    type: QualifiedType
key:
  - Offset
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/StructField.h"

class model::StructField : public model::generated::StructField {
public:
  using generated::StructField::StructField;

public:
  Identifier name() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/Model/Generated/Late/StructField.h"
