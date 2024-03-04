#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Model/UnionField.h"

/* TUPLE-TREE-YAML
name: UnionDefinition
doc: |
  A union type definition in the model.
  Unions are actually typedefs of unnamed unions in C.
type: struct
inherits: TypeDefinition
fields:
  - name: Fields
    sequence:
      type: SortedVector
      elementType: UnionField
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/UnionDefinition.h"

class model::UnionDefinition : public model::generated::UnionDefinition {
public:
  using generated::UnionDefinition::UnionDefinition;

  model::UnionField &addField(uint64_t Index, UpcastableType &&Type) {
    revng_assert(!Fields().contains(Index));
    model::UnionField &NewField = Fields()[Index];
    NewField.Type() = std::move(Type);
    return NewField;
  }
  model::UnionField &addField(UpcastableType &&Type) {
    return addField(Fields().size(), std::move(Type));
  }

public:
  const llvm::SmallVector<const model::Type *, 4> edges() const {
    llvm::SmallVector<const model::Type *> Result;

    for (auto &Field : Fields())
      if (!Field.Type().empty())
        Result.push_back(Field.Type().get());

    return Result;
  }
};

#include "revng/Model/Generated/Late/UnionDefinition.h"
