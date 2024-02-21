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
  static constexpr const char *AutomaticNamePrefix = "union_";

public:
  using generated::UnionDefinition::UnionDefinition;

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

#include "revng/Model/Generated/Late/UnionDefinition.h"
