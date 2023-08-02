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
fields:
  - name: PrimitiveKind
    type: PrimitiveTypeKind
  - name: Size
    doc: Size in bytes
    type: uint8_t
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/PrimitiveType.h"

class model::PrimitiveType : public model::generated::PrimitiveType {
public:
  using generated::PrimitiveType::PrimitiveType;

  // These constructors have to be overridden because primitive type IDs
  // are special: they are never randomly generated and instead depend on
  // the type specifics.
  explicit PrimitiveType(uint64_t ID);
  PrimitiveType(PrimitiveTypeKind::Values PrimitiveKind, uint8_t ByteSize);

public:
  static uint64_t FirstNonPrimitiveID;

public:
  Identifier name() const;

public:
  const llvm::SmallVector<model::QualifiedType, 4> edges() const { return {}; }

public:
  static bool classof(const Type *T) { return classof(T->key()); }
  static bool classof(const Key &K) { return std::get<1>(K) == AssociatedKind; }
};

namespace detail {
inline uint64_t getFirstNonPrimitiveID() {
  using namespace model;
  using namespace PrimitiveTypeKind;
  return PrimitiveType(static_cast<Values>(Count - 1), 16).ID() + 1;
}
} // namespace detail

inline uint64_t
  model::PrimitiveType::FirstNonPrimitiveID = detail::getFirstNonPrimitiveID();

#include "revng/Model/Generated/Late/PrimitiveType.h"
