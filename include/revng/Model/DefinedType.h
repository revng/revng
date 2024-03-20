#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Type.h"

/* TUPLE-TREE-YAML
name: DefinedType
type: struct
inherits: Type
fields:
  - name: Definition
    reference:
      pointeeType: TypeDefinition
      rootType: Binary
TUPLE-TREE-YAML */

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
  model::TypeDefinition &asDefinition() {
    revng_assert(Definition().isValid());
    return *Definition().get();
  }
  const model::TypeDefinition &asDefinition() const {
    revng_assert(Definition().isValid());
    return *Definition().get();
  }
};

#include "revng/Model/Generated/Late/DefinedType.h"
