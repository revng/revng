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
    type: Type
    upcastable: true
  - name: Entries
    sequence:
      type: SortedVector
      elementType: EnumEntry
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/EnumDefinition.h"

class model::EnumDefinition : public model::generated::EnumDefinition {
public:
  using generated::EnumDefinition::EnumDefinition;

public:
  Identifier entryName(const model::EnumEntry &Entry) const;

public:
  /// The helper for underlying type unwrapping.
  /// Use this when you need to access/modify the existing type,
  /// and \ref Type() when you need to assign a new one.
  model::PrimitiveType &underlyingType() {
    return UnderlyingType()->toPrimitive();
  }

  /// The helper for segment type unwrapping.
  /// Use this when you need to access/modify the existing struct,
  /// and \ref Type() when you need to assign a new one.
  const model::PrimitiveType &underlyingType() const {
    return UnderlyingType()->toPrimitive();
  }

public:
  llvm::SmallVector<const model::Type *, 4> edges() const {
    if (!UnderlyingType().empty())
      return { UnderlyingType().get() };
    else
      return {};
  }
  llvm::SmallVector<model::Type *, 4> edges() {
    if (!UnderlyingType().empty())
      return { UnderlyingType().get() };
    else
      return {};
  }
};

#include "revng/Model/Generated/Late/EnumDefinition.h"
