#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/EnumEntry.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeKind.h"

/* TUPLE-TREE-YAML
name: EnumType
doc: An enum type in model. Enums are actually typedefs of unnamed enums.
type: struct
inherits: Type
tag: Enum
fields:
  - name: UnderlyingType
    reference:
      pointeeType: model::Type
      rootType: model::Binary
  - name: Entries
    sequence:
      type: SortedVector
      elementType: model::EnumEntry
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/EnumType.h"

class model::EnumType : public model::generated::EnumType {
public:
  static constexpr const char *AutomaticNamePrefix = "enum_";
  static constexpr const TypeKind::Values AssociatedKind = TypeKind::Enum;

public:
  using generated::EnumType::EnumType;
  EnumType() : generated::EnumType() { Kind = AssociatedKind; }

public:
  Identifier name() const;

public:
  llvm::SmallVector<model::QualifiedType, 4> edges() {
    return { model::QualifiedType(UnderlyingType, {}) };
  }

public:
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<0>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/EnumType.h"
