#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/StructField.h"
#include "revng/Model/TypeDefinition.h"

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
