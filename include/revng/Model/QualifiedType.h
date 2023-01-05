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

// WIP: NEXT fix names. specifically rename Size, very misleading
/* TUPLE-TREE-YAML
name: QualifiedType
doc: A qualified version of a model::Type. Can have many nested qualifiers
type: struct
fields:
  - name: PrimitiveKind
    type: PrimitiveTypeKind
    optional: true
  - name: Size
    doc: Size in bytes
    type: uint8_t
    optional: true
  - name: UnqualifiedType
    reference:
      pointeeType: Type
      rootType: Binary
    optional: true
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
  /// If isPrimitive2 is true, returns that name of the primitive type
  Identifier primitiveName() const;

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

  bool empty() const { return not isPrimitive2() and not isTrull(); }

  // WIP: NEXT
  bool isPrimitive2() const {
    return PrimitiveKind() != PrimitiveTypeKind::Invalid;
  }

  bool isTrull() const {
    return not isPrimitive2() and not UnqualifiedType().empty();
  }

  bool isValid() const {
    return not(PrimitiveKind() == PrimitiveTypeKind::Invalid
               and not UnqualifiedType().isValid());
  }

public:
  [[nodiscard]] model::QualifiedType getPointerTo(model::Architecture::Values Arch) const {
    return addQualifier(model::Qualifier::createPointer(Arch));
  }

  [[nodiscard]] model::QualifiedType addQualifier(const model::Qualifier &Q) const {
    model::QualifiedType Result = *this;
    // WIP: NEXT reverse order
    Result.Qualifiers().insert(Result.Qualifiers().begin(), Q);
    return Result;
  }

  [[nodiscard]] model::QualifiedType popQualifier() const {
    model::QualifiedType Result = *this;
    Result.Qualifiers().erase(Result.Qualifiers().begin());
    return Result;
  }

  [[nodiscard]] model::QualifiedType replaceQualifiers(
  const std::vector<model::Qualifier> &NewQualifiers) const {
    revng_assert(not empty());
    model::QualifiedType Result = *this;
    Result.Qualifiers() = NewQualifiers;
    return Result;
  }

public:
  static model::QualifiedType
  getPrimitiveType(PrimitiveTypeKind::Values V, uint8_t ByteSize) {
    model::QualifiedType Result;
    Result.PrimitiveKind() = V;
    Result.Size() = ByteSize;
    return Result;
  }

  static model::QualifiedType getVoidType() {
    return getPrimitiveType(PrimitiveTypeKind::Void, 0);
  }

  // WIP: NEXT rename
  static model::QualifiedType getLel(const model::TypePath &P) {
    model::QualifiedType Result;
    Result.UnqualifiedType() = P;
    return Result;
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;

  /// Returns a pseudo C representation of the type
  ///
  /// \note This is for debug purposes only. It's not a valid C type.
  std::string pseudoC() const debug_function;

  std::string getPrimitiveName() const;

  void dump() const debug_function;

  bool operator==(const QualifiedType &) const = default;
  std::strong_ordering operator<=>(const QualifiedType &Other) const {
    if (PrimitiveKind() != Other.PrimitiveKind())
      return PrimitiveKind() <=> Other.PrimitiveKind();

    if (Size() != Other.Size())
      return Size() <=> Other.Size();

    if (Qualifiers() < Other.Qualifiers())
      return std::strong_ordering::less;
    else if (Qualifiers() > Other.Qualifiers())
      return std::strong_ordering::greater;

    return UnqualifiedType() <=> Other.UnqualifiedType();
  }
};

#include "revng/Model/Generated/Late/QualifiedType.h"
