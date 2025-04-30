#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Type.h"

#include "revng/Model/Generated/Early/DefinedType.h"

class model::DefinedType : public model::generated::DefinedType {
public:
  using generated::DefinedType::DefinedType;

  static UpcastableType make(TypeOfDefinition Definition) {
    return UpcastableType::make<DefinedType>(false, std::move(Definition));
  }

  static UpcastableType makeConst(TypeOfDefinition Definition) {
    return UpcastableType::make<DefinedType>(true, std::move(Definition));
  }

public:
  model::TypeDefinition &unwrap() {
    revng_assert(Definition().isValid());
    return *Definition().get();
  }
  const model::TypeDefinition &unwrap() const {
    revng_assert(Definition().isValid());
    return *Definition().getConst();
  }
};

#include "revng/Model/Generated/Late/DefinedType.h"
