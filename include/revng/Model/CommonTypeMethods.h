#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/PrimitiveKind.h"

namespace model {
class VerifyHelper;

class TypeDefinition;
class TypedefDefinition;
class EnumDefinition;
class StructDefinition;
class UnionDefinition;

class Type;
class ArrayType;
class DefinedType;
class PointerType;
class PrimitiveType;

/// A common place to gather a bunch of generic type system helpers.
///
/// @tparam CRTP as of now, only `model::Type` and `model::TypeDefinition`
///              are supported.
template<typename CRTP>
class CommonTypeMethods {
public:
  /// Recursively computes size of the type.
  ///
  /// It asserts in cases where the size cannot be computed, for example, when
  /// the type system loops and the type's size depends on the type itself.
  ///
  /// \returns * `std::nullopt` if the type does not have size (for example,
  ///            it's a `void` primitive or a function type),
  ///          * size in bytes otherwise.
  std::optional<uint64_t> size() const debug_function;
  RecursiveCoroutine<std::optional<uint64_t>> size(VerifyHelper &VH) const;

public:
  ///
  /// \name Type classification helpers
  ///
  /// One set of these is provided for each major type kind. They provide
  /// an easy way to conditionally unwrap a type (`getX`) or check if it can be
  /// unwrapped (`isX`) or unwrap the type while asserting it is what you
  /// expect it to be.
  ///
  /// \note these helpers skip typedefs, if that's not what you want, use llvm
  ///       machinery instead (`dyn_cast`/`isa`)
  ///
  /// @{

  model::TypeDefinition *getPrototype();
  const model::TypeDefinition *getPrototype() const;
  bool isPrototype() const { return getPrototype() != nullptr; }
  model::TypeDefinition &toPrototype() {
    if (model::TypeDefinition *Result = getPrototype())
      return *Result;
    else
      revng_abort("Not a prototype!");
  }
  const model::TypeDefinition &toPrototype() const {
    if (const model::TypeDefinition *Result = getPrototype())
      return *Result;
    else
      revng_abort("Not a prototype!");
  }

  model::StructDefinition *getStruct();
  const model::StructDefinition *getStruct() const;
  bool isStruct() const { return getStruct() != nullptr; }
  model::StructDefinition &toStruct() {
    if (model::StructDefinition *Result = getStruct())
      return *Result;
    else
      revng_abort("Not a struct!");
  }
  const model::StructDefinition &toStruct() const {
    if (const model::StructDefinition *Result = getStruct())
      return *Result;
    else
      revng_abort("Not a struct!");
  }

  model::UnionDefinition *getUnion();
  const model::UnionDefinition *getUnion() const;
  bool isUnion() const { return getUnion() != nullptr; }
  model::UnionDefinition &toUnion() {
    if (model::UnionDefinition *Result = getUnion())
      return *Result;
    else
      revng_abort("Not a union!");
  }
  const model::UnionDefinition &toUnion() const {
    if (const model::UnionDefinition *Result = getUnion())
      return *Result;
    else
      revng_abort("Not a union!");
  }

  model::EnumDefinition *getEnum();
  const model::EnumDefinition *getEnum() const;
  bool isEnum() const { return getEnum() != nullptr; }
  model::EnumDefinition &toEnum() {
    if (model::EnumDefinition *Result = getEnum())
      return *Result;
    else
      revng_abort("Not a enum!");
  }
  const model::EnumDefinition &toEnum() const {
    if (const model::EnumDefinition *Result = getEnum())
      return *Result;
    else
      revng_abort("Not a enum!");
  }

  model::PrimitiveType *getPrimitive();
  const model::PrimitiveType *getPrimitive() const;
  bool isPrimitive() const { return getPrimitive() != nullptr; }
  model::PrimitiveType &toPrimitive() {
    if (model::PrimitiveType *Result = getPrimitive())
      return *Result;
    else
      revng_abort("Not a primitive!");
  }
  const model::PrimitiveType &toPrimitive() const {
    if (const model::PrimitiveType *Result = getPrimitive())
      return *Result;
    else
      revng_abort("Not a primitive!");
  }

  model::ArrayType *getArray();
  const model::ArrayType *getArray() const;
  bool isArray() const { return getArray() != nullptr; }
  model::ArrayType &toArray() {
    if (model::ArrayType *Result = getArray())
      return *Result;
    else
      revng_abort("Not an array!");
  }
  const model::ArrayType &toArray() const {
    if (const model::ArrayType *Result = getArray())
      return *Result;
    else
      revng_abort("Not an array!");
  }

  model::PointerType *getPointer();
  const model::PointerType *getPointer() const;
  bool isPointer() const { return getPointer() != nullptr; }
  model::PointerType &toPointer() {
    if (model::PointerType *Result = getPointer())
      return *Result;
    else
      revng_abort("Not a pointer!");
  }
  const model::PointerType &toPointer() const {
    if (const model::PointerType *Result = getPointer())
      return *Result;
    else
      revng_abort("Not a pointer!");
  }

  /// @}

public:
  /// This helper returns true if and only if this type is an object from
  /// the C++ point of view, as in anything that can have size.
  ///  ///
  /// \note similarly to other `isX` methods, this also skips typedefs.
  bool isObject() const { return not isVoidPrimitive() and not isPrototype(); }

  /// This helper returns true if and only if this type is a primitive or
  /// a pointer.
  ///
  /// \note This helpers considers enums to be primitives.
  ///
  /// \note it asserts if called on something that's not an object.
  ///
  /// \note similarly to other `isX` methods, this also skips typedefs.
  bool isScalar() const;

  /// This helper returns true if and only if this type is not scalar.
  ///
  /// \note it asserts if called on something that's not an object.
  ///
  /// \note similarly to other `isX` methods, this also skips typedefs.
  bool isAggregate() const { return !isScalar(); }

public:
  /// Use this method to check whether a specific type is constant or not, as
  /// opposed to the \ref IsConst() member (which should be reserved for
  /// assigning `const`ness), since it unwraps typedefs and arrays.
  bool isConst() const;

public:
  /// Returns the definition type if and only if it's a defined type.
  ///
  /// \note This returns `nullptr` if the type is not a definition, for example
  ///       if it's a primitive, pointer, array and so on.
  ///
  /// \note This ignores the constness BUT it does NOT skip typedefs.
  model::TypeDefinition *tryGetAsDefinition();
  const model::TypeDefinition *tryGetAsDefinition() const;

public:
  /// Returns the pointee type while unwrapping typedefs.
  ///
  /// \note This asserts if the type is *not* a pointer, so only use it when
  ///       it's guaranteed to be one.
  ///
  /// \note This ignores the constness of the dropped pointer (and typedefs
  ///       in the way)
  model::Type &getPointee() { return *toPointer().PointeeType(); }
  const model::Type &getPointee() const { return *toPointer().PointeeType(); }

  /// Returns the element type of an array type while unwrapping typedefs.
  ///
  /// \note This asserts if the type is _not_ an array, so only use it when
  ///       it's guaranteed to be one.
  ///
  /// \note This ignores the constness of the dropped array (and typedefs
  ///       in the way)
  model::Type &getArrayElement() { return *toArray().ElementType(); }
  const model::Type &getArrayElement() const {
    return *toArray().ElementType();
  }

private:
  using PKind = model::PrimitiveKind::Values;
  constexpr static auto PoN = model::PrimitiveKind::PointerOrNumber;

public:
  ///
  /// \name Primitive classification helpers
  ///
  /// These help semantically checking whether a type given type is a primitive
  /// of a given type.
  ///
  /// \note Think of these as a specialized extension of
  ///       the "Type classification helpers" group with only `ifX` provided.
  ///
  /// \note these helpers skip typedefs, if that's not what you want, use llvm
  ///       machinery instead (`dyn_cast`/`isa`)
  ///
  /// @{

  bool isPrimitive(model::PrimitiveKind::Values Kind) const;
  bool isVoidPrimitive() const { return isPrimitive(PKind::Void); }
  bool isGenericPrimitive() const { return isPrimitive(PKind::Generic); }
  bool isPointerOrNumberPrimitive() const { return isPrimitive(PoN); }
  bool isNumberPrimitive() const { return isPrimitive(PKind::Number); }
  bool isUnsignedPrimitive() const { return isPrimitive(PKind::Unsigned); }
  bool isSignedPrimitive() const { return isPrimitive(PKind::Signed); }
  bool isFloatPrimitive() const { return isPrimitive(PKind::Float); }

  /// @}

public:
  /// The helper for typedef unwrapping.
  ///
  /// \returns - nullptr if called with a non-typedef definition, since it's not
  ///            possible to produce a `model::Type` for one of those without
  ///            placing it into a binary.
  ///          - a pointer to an unwrapped `model::Type` in all the other cases.
  model::Type *skipTypedefs();
  const model::Type *skipTypedefs() const;

  /// The helper for unwrapping typedefs, including const ones.
  /// See \ref skipTypedef documentation for specifics.
  model::Type *skipConstAndTypedefs();
  const model::Type *skipConstAndTypedefs() const;

public:
  /// Use this method to check whether a specific type is a typedef or not.
  ///
  /// \note This is not a part of the "Type classification helpers" group as
  ///       it doesn't make sense for it to skip typedefs.
  bool isTypedef() const {
    const model::TypeDefinition *Definition = tryGetAsDefinition();
    return Definition && llvm::isa<model::TypedefDefinition>(Definition);
  }

private:
  /// The CRTP unwrapping helper.
  CRTP &get() { return static_cast<CRTP &>(*this); }
  const CRTP &get() const { return static_cast<const CRTP &>(*this); }
};

} // namespace model
