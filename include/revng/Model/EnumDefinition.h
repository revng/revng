#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/EnumEntry.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/TypeDefinition.h"

/* TUPLE-TREE-YAML
name: EnumDefinition
doc: |
  An enum type definition in the model.
  Enums are actually typedefs of unnamed enums.
type: struct
inherits: TypeDefinition
fields:
  - name: UnderlyingType
    type: QualifiedType
  - name: Entries
    sequence:
      type: SortedVector
      elementType: EnumEntry
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/EnumDefinition.h"

class model::EnumDefinition : public model::generated::EnumDefinition {
public:
  static constexpr const char *AutomaticNamePrefix = "enum_";

public:
  using generated::EnumDefinition::EnumDefinition;

public:
  Identifier name() const;
  Identifier entryName(const model::EnumEntry &Entry) const;

public:
  const llvm::SmallVector<model::QualifiedType, 4> edges() const {
    return { UnderlyingType() };
  }

public:
  static bool classof(const TypeDefinition *D) { return classof(D->key()); }
  static bool classof(const Key &K) { return std::get<1>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/EnumDefinition.h"
