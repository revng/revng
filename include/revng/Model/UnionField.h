#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/Type.h"

/* TUPLE-TREE-YAML
name: UnionField
doc: |-
  An alternative of a `UnionDefinition`.

  It is composed by a index and the `Type`.
type: struct
fields:
  - name: Index
    type: uint64_t
    doc: The index of the alternative within the `union`.
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
    doc: The type of this `union` alternative.
    upcastable: true
key:
  - Index
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/UnionField.h"

namespace model {
class VerifyHelper;
}

class model::UnionField : public model::generated::UnionField {
public:
  using generated::UnionField::UnionField;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/UnionField.h"
