#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/EnumEntry.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/Type.h"

/* TUPLE-TREE-YAML
name: EnumType
doc: An enum type in model. Enums are actually typedefs of unnamed enums.
type: struct
inherits: Type
fields:
  - name: UnderlyingType
    type: QualifiedType
  - name: Entries
    sequence:
      type: SortedVector
      elementType: EnumEntry
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/EnumType.h"

class model::EnumType : public model::generated::EnumType {
public:
  static constexpr const char *AutomaticNamePrefix = "enum_";

public:
  using generated::EnumType::EnumType;
  EnumType() : generated::EnumType() {}

public:
  Identifier name() const;
  Identifier entryName(const model::EnumEntry &Entry) const;

public:
  const llvm::SmallVector<model::QualifiedType, 4> edges() const {
    return { UnderlyingType() };
  }

public:
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<1>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/EnumType.h"
