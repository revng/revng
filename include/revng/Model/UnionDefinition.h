#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/TypeDefinition.h"
#include "revng/Model/UnionField.h"

#include "revng/Model/Generated/Early/UnionDefinition.h"

class model::UnionDefinition : public model::generated::UnionDefinition {
public:
  using generated::UnionDefinition::UnionDefinition;

  model::UnionField &addField(UpcastableType &&Type) {
    revng_assert(!Fields().contains(Fields().size()));
    model::UnionField &NewField = Fields()[Fields().size()];
    NewField.Type() = std::move(Type);
    return NewField;
  }

public:
  llvm::SmallVector<const model::Type *, 4> edges() const {
    llvm::SmallVector<const model::Type *> Result;

    for (const auto &Field : Fields())
      if (!Field.Type().isEmpty())
        Result.push_back(Field.Type().get());

    return Result;
  }
  llvm::SmallVector<model::Type *, 4> edges() {
    llvm::SmallVector<model::Type *> Result;

    for (auto &Field : Fields())
      if (!Field.Type().isEmpty())
        Result.push_back(Field.Type().get());

    return Result;
  }
};

#include "revng/Model/Generated/Late/UnionDefinition.h"
