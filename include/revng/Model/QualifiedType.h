#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <optional>

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/PrimitiveTypeKind.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/TypeKind.h"
#include "revng/Model/VerifyHelper.h"

/* TUPLE-TREE-YAML
name: QualifiedType
doc: A qualified version of a model::Type. Can have many nested qualifiers
type: struct
fields:
  - name: UnqualifiedType
    reference:
      pointeeType: Type
      rootType: Binary
  - name: Qualifiers
    sequence:
      type: "std::vector"
      elementType: Qualifier
    optional: true
TUPLE-TREE-YAML */

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
  bool isPrimitive(model::PrimitiveTypeKind::Values V) const;
  /// Checks if is float, unwrapping typedefs
  bool isFloat() const { return isPrimitive(model::PrimitiveTypeKind::Float); }
  /// Checks if is void, unwrapping typedefs
  bool isVoid() const { return isPrimitive(model::PrimitiveTypeKind::Void); }
  /// Checks if is an array type, unwrapping typedefs
  bool isArray() const;
  /// Checks if is a pointer type, unwrapping typedefs
  bool isPointer() const;
  /// Checks if is a const type, unwrapping typedefs
  bool isConst() const;
  /// Checks if is of a given TypeKind, unwrapping typedefs
  bool is(model::TypeKind::Values K) const;

  /// If this QualifiedType is a Typedef with no qualifiers, unwrap the typedef
  /// and repeat
  model::QualifiedType skipTypedefs() const;

  /// If this QualifiedType is a function type, return it skipping over typedefs
  std::optional<model::TypePath> getFunctionType() const;

  static std::optional<model::TypePath>
  getFunctionType(const model::TypePath &TypePath) {
    return model::QualifiedType{ TypePath, {} }.getFunctionType();
  }

public:
  model::QualifiedType getPointerTo(model::Architecture::Values Arch) const {
    QualifiedType Result = *this;
    Result.Qualifiers().insert(Result.Qualifiers().begin(),
                               model::Qualifier::createPointer(Arch));
    return Result;
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
  void dump() const debug_function;

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
