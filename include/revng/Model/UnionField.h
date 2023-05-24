#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: UnionField
doc: A field of a union type in model, with position, qualified type, and name
type: struct
fields:
  - name: Index
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
  - Index
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/UnionField.h"

class model::UnionField : public model::generated::UnionField {
public:
  using generated::UnionField::UnionField;

public:
  Identifier name() const;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;
};

#include "revng/Model/Generated/Late/UnionField.h"
