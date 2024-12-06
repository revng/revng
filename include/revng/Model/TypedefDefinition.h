#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeDefinition.h"

/* TUPLE-TREE-YAML
name: TypedefDefinition
doc: |-
  A `typedef` type.

  Note, that unlike in C, two `typedef`s aliasing the same type are actually
  distinct types.
type: struct
inherits: TypeDefinition
fields:
  - name: UnderlyingType
    type: Type
    doc: The type this `typedef` is aliasing.
    upcastable: true
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/TypedefDefinition.h"

class model::TypedefDefinition : public model::generated::TypedefDefinition {
public:
  using generated::TypedefDefinition::TypedefDefinition;
  explicit TypedefDefinition(UpcastableType &&Underlying) :
    generated::TypedefDefinition() {

    UnderlyingType() = std::move(Underlying);
  }

public:
  llvm::SmallVector<const model::Type *, 4> edges() const {
    if (!UnderlyingType().isEmpty())
      return { UnderlyingType().get() };
    else
      return {};
  }
  llvm::SmallVector<model::Type *, 4> edges() {
    if (!UnderlyingType().isEmpty())
      return { UnderlyingType().get() };
    else
      return {};
  }
};

#include "revng/Model/Generated/Late/TypedefDefinition.h"
