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
  A `struct` type.
type: struct
inherits: TypeDefinition
fields:
  - name: Size
    doc: The size, in bytes, of the `struct`.
    type: uint64_t
  - name: CanContainCode
    doc: |
      When this field is set to `true` *and* this `struct` is reachable for
      a segment's root type without traversing pointer, arrays or other
      qualifiers, the padding of the struct is treated at if it contains code.
    type: bool
    optional: true
  - name: Fields
    doc: The list of fields of this `struct`.
    sequence:
      type: SortedVector
      elementType: StructField
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/StructDefinition.h"

class model::StructDefinition : public model::generated::StructDefinition {
public:
  using generated::StructDefinition::StructDefinition;
  explicit StructDefinition(uint64_t StructSize) :
    generated::StructDefinition() {

    Size() = StructSize;
  }

  StructField &addField(uint64_t Offset, UpcastableType &&Type) {
    auto &&[Iterator, Success] = Fields().emplace(Offset);
    revng_assert(Success);
    Iterator->Type() = std::move(Type);
    return *Iterator;
  }

public:
  llvm::SmallVector<const model::Type *, 4> edges() const {
    llvm::SmallVector<const model::Type *, 4> Result;

    for (const auto &Field : Fields())
      if (!Field.Type().isEmpty())
        Result.push_back(Field.Type().get());

    return Result;
  }
  llvm::SmallVector<model::Type *, 4> edges() {
    llvm::SmallVector<model::Type *, 4> Result;

    for (auto &Field : Fields())
      if (!Field.Type().isEmpty())
        Result.push_back(Field.Type().get());

    return Result;
  }
};

#include "revng/Model/Generated/Late/StructDefinition.h"
