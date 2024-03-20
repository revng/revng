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
  - name: Fields
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
    auto [Iterator, Success] = Fields().emplace(Offset);
    revng_assert(Success);
    Iterator->Type() = std::move(Type);
    return *Iterator;
  }

public:
  const llvm::SmallVector<const model::Type *, 4> edges() const {
    llvm::SmallVector<const model::Type *, 4> Result;

    for (auto &Field : Fields())
      if (!Field.Type().empty())
        Result.push_back(Field.Type().get());

    return Result;
  }
};

#include "revng/Model/Generated/Late/StructDefinition.h"
