#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: StructField
doc: |-
  A field of a `StructDefinition`.

  It is composed by the offset of the field and its type.
type: struct
fields:
  - name: Offset
    type: uint64_t
    doc: Offset at which the field starts within the `struct`.
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
    type: Type
    doc: The type of the field.
    upcastable: true
key:
  - Offset
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/StructField.h"

class model::StructField : public model::generated::StructField {
public:
  using generated::StructField::StructField;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/StructField.h"
