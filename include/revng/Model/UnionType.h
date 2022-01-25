#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeKind.h"
#include "revng/Model/UnionField.h"

/* TUPLE-TREE-YAML
name: UnionType
doc: |
  A union type in model.
  Unions are actually typedefs of unnamed unions in C.
type: struct
inherits: Type
tag: Union
fields:
  - name: Fields
    sequence:
      type: SortedVector
      elementType: model::UnionField
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/UnionType.h"

class model::UnionType : public model::generated::UnionType {
public:
  static constexpr const char *AutomaticNamePrefix = "union_";
  static constexpr const TypeKind::Values AssociatedKind = TypeKind::Union;

public:
  using generated::UnionType::UnionType;
  UnionType() : generated::UnionType() { Kind = AssociatedKind; }

public:
  Identifier name() const;

public:
  llvm::SmallVector<model::QualifiedType, 4> edges() {
    llvm::SmallVector<model::QualifiedType, 4> Result;

    for (auto &Field : Fields)
      Result.push_back(Field.Type);

    return Result;
  }

public:
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<0>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/UnionType.h"
