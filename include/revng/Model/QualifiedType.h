#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <optional>

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/PrimitiveKind.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/TypeDefinitionKind.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: QualifiedType
doc: |
  A qualified version of a model::TypeDefinition. Can have many nested
  qualifiers
type: struct
fields:
  - name: UnqualifiedType
    reference:
      pointeeType: TypeDefinition
      rootType: Binary
  - name: Qualifiers
    sequence:
      type: "std::vector"
      elementType: Qualifier
    optional: true
TUPLE-TREE-YAML */

namespace model {
using DefinitionReference = TupleTreeReference<model::TypeDefinition,
                                               model::Binary>;
}

#include "revng/Model/Generated/Early/QualifiedType.h"

class model::QualifiedType : public model::generated::QualifiedType {
public:
  using generated::QualifiedType::QualifiedType;

public:
  std::optional<uint64_t> size() const debug_function;
  RecursiveCoroutine<std::optional<uint64_t>> size(VerifyHelper &VH) const;

  std::optional<uint64_t> trySize() const debug_function;
  RecursiveCoroutine<std::optional<uint64_t>> trySize(VerifyHelper &VH) const;

public:
  /// Checks if is a scalar type, unwrapping typedefs
  bool isScalar() const;
  /// Checks if is a primitive type, unwrapping typedefs
  bool isPrimitive() const;
  /// Checks if is a primitive type of a specific kind, unwrapping typedefs
  bool isPrimitive(model::PrimitiveKind::Values V) const;
  /// Checks if is float, unwrapping typedefs
  bool isFloat() const { return isPrimitive(model::PrimitiveKind::Float); }
  /// Checks if is void, unwrapping typedefs
  bool isVoid() const { return isPrimitive(model::PrimitiveKind::Void); }
  /// Checks if is an array type, unwrapping typedefs
  bool isArray() const;
  /// Checks if is a pointer type, unwrapping typedefs
  bool isPointer() const;
  /// Checks if is a const type, unwrapping typedefs
  bool isConst() const;
  /// Checks if is of a given TypeDefinitionKind, unwrapping typedefs
  bool is(model::TypeDefinitionKind::Values K) const;

  /// If this QualifiedType is a Typedef with no qualifiers, unwrap the typedef
  /// and repeat
  model::QualifiedType skipTypedefs() const;

  /// If this QualifiedType is a function type, return it skipping over typedefs
  std::optional<model::DefinitionReference> getFunctionType() const;

  static std::optional<model::DefinitionReference>
  getFunctionType(const model::DefinitionReference &Path) {
    return model::QualifiedType{ Path, {} }.getFunctionType();
  }

public:
  model::QualifiedType getPointerTo(model::Architecture::Values Arch) const;

  model::QualifiedType stripPointer() const {
    model::QualifiedType Result = *this;
    revng_assert(not Result.Qualifiers().empty()
                 and model::Qualifier::isPointer(Result.Qualifiers().front()));
    Result.Qualifiers().erase(Result.Qualifiers().begin());
    return Result;
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;

  bool operator==(const QualifiedType &) const = default;
  std::strong_ordering operator<=>(const QualifiedType &Other) const {
    if (Qualifiers() < Other.Qualifiers())
      return std::strong_ordering::less;

    if (Qualifiers() > Other.Qualifiers())
      return std::strong_ordering::greater;

    return UnqualifiedType() <=> Other.UnqualifiedType();
  }
};

#include "revng/Model/Generated/Late/QualifiedType.h"
