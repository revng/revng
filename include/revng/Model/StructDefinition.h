#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/StructField.h"
#include "revng/Model/TypeDefinition.h"

/* TUPLE-TREE-YAML
name: StructDefinition
doc: |
  A struct type in model.
  Structs are actually typedefs of unnamed structs in C.
type: struct
inherits: TypeDefinition
fields:
  - name: Size
    doc: Size in bytes
    type: uint64_t
  - name: CanContainCode
    doc: Whether this struct, when in a Segment can contain code.
    type: bool
    optional: true
  - name: Fields
    sequence:
      type: SortedVector
      elementType: StructField
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/StructDefinition.h"

class model::StructDefinition : public model::generated::StructDefinition {
public:
  static constexpr const char *AutomaticNamePrefix = "struct_";

public:
  using generated::StructDefinition::StructDefinition;

public:
  Identifier name() const;

public:
  const llvm::SmallVector<model::QualifiedType, 4> edges() const {
    llvm::SmallVector<model::QualifiedType, 4> Result;

    for (auto &Field : Fields())
      Result.push_back(Field.Type());

    return Result;
  }
};

#include "revng/Model/Generated/Late/StructDefinition.h"
