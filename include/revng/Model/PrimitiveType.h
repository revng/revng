#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/Type.h"

/* TUPLE-TREE-YAML
name: PrimitiveType
doc: "A primitive type in model: sized integers, booleans, floats and void"
type: struct
inherits: Type
tag: Primitive
fields:
  - name: PrimitiveKind
    type: model::PrimitiveTypeKind::Values
  - name: Size
    doc: Size in bytes
    type: uint8_t
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/PrimitiveType.h"

class model::PrimitiveType : public model::generated::PrimitiveType {
public:
  static constexpr const TypeKind::Values AssociatedKind = TypeKind::Primitive;

public:
  using generated::PrimitiveType::PrimitiveType;
  // TODO: these do not conform to the constructors convention
  PrimitiveType() : generated::PrimitiveType() { Kind = AssociatedKind; };
  explicit PrimitiveType(uint64_t ID);
  PrimitiveType(PrimitiveTypeKind::Values PrimitiveKind, uint8_t ByteSize);

public:
  Identifier name() const;

public:
  llvm::SmallVector<model::QualifiedType, 4> edges() { return {}; }

public:
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<0>(K) == AssociatedKind; }
};

#include "revng/Model/Generated/Late/PrimitiveType.h"
