#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Identifier.h"
#include "revng/Model/PrimitiveKind.h"
#include "revng/Model/TypeDefinition.h"

/* TUPLE-TREE-YAML
name: PrimitiveDefinition
doc: "A primitive type in model: sized integers, booleans, floats and void"
type: struct
inherits: TypeDefinition
fields:
  - name: PrimitiveKind
    type: PrimitiveKind
  - name: Size
    doc: Size in bytes
    type: uint8_t
TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/PrimitiveDefinition.h"

class model::PrimitiveDefinition
  : public model::generated::PrimitiveDefinition {
public:
  using generated::PrimitiveDefinition::PrimitiveDefinition;

  // These constructors have to be overridden because primitive type IDs
  // are special: they are never randomly generated and instead depend on
  // the type specifics.
  explicit PrimitiveDefinition(uint64_t ID);
  PrimitiveDefinition(PrimitiveKind::Values PrimitiveKind, uint8_t ByteSize);

  static std::optional<model::PrimitiveDefinition>
  fromName(llvm::StringRef Name);

public:
  static uint64_t FirstNonPrimitiveID;

public:
  Identifier name() const;

public:
  const llvm::SmallVector<model::QualifiedType, 4> edges() const { return {}; }

public:
  static bool classof(const TypeDefinition *D) { return classof(D->key()); }
  static bool classof(const Key &K) { return std::get<1>(K) == AssociatedKind; }
};

namespace detail {
inline uint64_t firstNonPrimitiveID() {
  using PKind = model::PrimitiveKind::Values;
  using PrimitiveDefinition = model::PrimitiveDefinition;
  return PrimitiveDefinition(static_cast<PKind>(PKind::Count - 1), 16).ID() + 1;
}

using PD = model::PrimitiveDefinition;
} // namespace detail

inline uint64_t detail::PD::FirstNonPrimitiveID = detail::firstNonPrimitiveID();

#include "revng/Model/Generated/Late/PrimitiveDefinition.h"
