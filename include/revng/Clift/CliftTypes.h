#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "revng/Clift/CliftAttributes.h"
#include "revng/Clift/CliftDialect.h"
#include "revng/Clift/CliftInterfaces.h"
#include "revng/Support/Assert.h"

// This include should stay here for correct build procedure
#define GET_TYPEDEF_CLASSES
#include "revng/Clift/CliftTypes.h.inc"

namespace clift {

//===------------------------------ Typedefs ------------------------------===//

struct TypedefDecomposition {
  mlir::Type Type;
  bool HasConstTypedef;
};

/// Recursively decomposes a typedef into its underlying non-typedef type and a
/// boolean indicating whether any of the typedefs added const. Note that the
/// underlying type itself may also be const while the boolean may be false.
TypedefDecomposition decomposeTypedef(mlir::Type Type);

/// Recursively unwraps typedefs and returns the underlying non-typedef type
/// unchanged.
[[nodiscard]] mlir::Type unwrapTypedefs(mlir::Type Type);

/// Recursively unwraps typedefs and returns the underlying non-typedef type
/// unchanged.
[[nodiscard]] inline AddressableType unwrapTypedefs(AddressableType Type) {
  return mlir::cast<AddressableType>(unwrapTypedefs(mlir::Type(Type)));
}

/// Recursively unwraps typedefs and returns the underlying non-typedef type,
/// with any qualifiers from wrapping typedefs added onto the resulting type.
[[nodiscard]] mlir::Type collapseTypedefs(mlir::Type Type);

/// Recursively unwraps typedefs and returns the underlying non-typedef type,
/// with any qualifiers from wrapping typedefs added onto the resulting type.
[[nodiscard]] inline AddressableType collapseTypedefs(AddressableType Type) {
  return mlir::cast<AddressableType>(collapseTypedefs(mlir::Type(Type)));
}

//===----------------------------- Type casts -----------------------------===//

template<std::derived_from<mlir::Type>... ToTypes,
         std::derived_from<mlir::Type> FromType>
[[nodiscard]] bool unwrapped_isa(FromType From) {
  static_assert(not(std::is_same_v<ToTypes, TypedefType> || ...),
                "TypedefType would never be matched after unwrapping.");

  // TODO: Find out if there is a way to detect up-casts at compile time.
  return mlir::isa<ToTypes...>(clift::unwrapTypedefs(From));
}

template<std::derived_from<mlir::Type>... ToTypes,
         std::derived_from<mlir::Type> FromType>
[[nodiscard]] bool unwrapped_isa_and_present(FromType From) {
  return From and clift::unwrapped_isa<ToTypes...>(From);
}

template<std::derived_from<mlir::Type> ToType,
         std::derived_from<mlir::Type> FromType>
[[nodiscard]] ToType unwrapped_cast(FromType From) {
  static_assert(not std::is_same_v<ToType, TypedefType>,
                "TypedefType would never be matched after unwrapping.");

  // TODO: Find out if there is a way to detect up-casts at compile time.
  return mlir::cast<ToType>(clift::unwrapTypedefs(From));
}

template<std::derived_from<mlir::Type> ToType,
         std::derived_from<mlir::Type> FromType>
[[nodiscard]] ToType unwrapped_cast_if_present(FromType From) {
  return From ? clift::unwrapped_cast<ToType>(From) : ToType();
}

template<std::derived_from<mlir::Type> ToType,
         std::derived_from<mlir::Type> FromType>
[[nodiscard]] ToType unwrapped_dyn_cast(FromType From) {
  static_assert(not std::is_same_v<ToType, TypedefType>,
                "TypedefType would never be matched after unwrapping.");

  // TODO: Find out if there is a way to detect up-casts at compile time.
  return mlir::dyn_cast<ToType>(clift::unwrapTypedefs(From));
}

template<std::derived_from<mlir::Type> ToType,
         std::derived_from<mlir::Type> FromType>
[[nodiscard]] ToType unwrapped_dyn_cast_if_present(FromType From) {
  return From ? clift::unwrapped_dyn_cast<ToType>(From) : ToType();
}

//===---------------------------- CV-Qualifiers ---------------------------===//

/// Determine if the type is top-level const qualified.
[[nodiscard]] bool isConst(mlir::Type Type);

/// Add top-level qualification to the given type, if it is an AddressableType.
/// Otherwise returns the type unchanged.
[[nodiscard]] mlir::Type addConst(mlir::Type Type);

/// Add top-level qualification to the given type, if it is an AddressableType.
/// Otherwise returns the type unchanged.
template<typename TypeT>
[[nodiscard]] TypeT addConst(TypeT Type) {
  return mlir::cast<TypeT>(addConst(mlir::Type(Type)));
}

/// Remove top-level qualification from the given type, if it is an
/// AddressableType. Otherwise returns the type unchanged.
[[nodiscard]] mlir::Type removeConst(mlir::Type Type);

/// Remove top-level qualification from the given type, if it is an
/// AddressableType. Otherwise returns the type unchanged.
template<typename TypeT>
[[nodiscard]] TypeT removeConst(TypeT Type) {
  return mlir::cast<TypeT>(removeConst(mlir::Type(Type)));
}

/// Determine if the two types are equivalent, ignoring Clift CV-qualifiers.
[[nodiscard]] bool equivalent(mlir::Type Lhs, mlir::Type Rhs);

/// Determine if the two types are equivalent after unwrapping typedefs and
/// ignoring Clift CV-qualifiers.
[[nodiscard]] inline bool equivalentUnwrapped(mlir::Type Lhs, mlir::Type Rhs) {
  return equivalent(unwrapTypedefs(Lhs), unwrapTypedefs(Rhs));
}

//===-------------------------- Object type size --------------------------===//

/// Returns the size of the given type, assuming it is an ObjectType (after
/// unwrapping typedefs).
[[nodiscard]] uint64_t getObjectSize(mlir::Type Type);

/// Returns the size of the given value, assuming its type is an ObjectType
/// (after unwrapping typedefs).
[[nodiscard]] inline uint64_t getObjectSize(mlir::Value Value) {
  return getObjectSize(Value.getType());
}

/// Returns the size of the given type, if it is an ObjectType (after unwrapping
/// typedefs). Otherwise returns zero.
[[nodiscard]] uint64_t getObjectSizeOrZero(mlir::Type Type);

/// Returns the size of the given value, if its type is an ObjectType (after
/// unwrapping typedefs).
[[nodiscard]] inline uint64_t getObjectSizeOrZero(mlir::Value Value) {
  return getObjectSize(Value.getType());
}

/// Returns the bit width of the given type, assuming it is an IntegralType
/// (after unwrapping typedefs).
[[nodiscard]] inline uint64_t getBitWidth(mlir::Type Type) {
  revng_assert(clift::unwrapped_isa<IntegralType>(Type));
  return getObjectSize(Type) * 8;
}

/// Returns the bit width of the given value, assuming its type is an
/// IntegralType (after unwrapping typedefs).
[[nodiscard]] inline uint64_t getBitWidth(mlir::Value Value) {
  return getBitWidth(Value.getType());
}

//===--------------------------- Integral types ---------------------------===//

/// Returns true if the specified type is a signed integer type (including
/// enum with a signed underlying type), or a typedef for such a type.
bool isSigned(mlir::Type Type);

/// Returns true if the specified type is an unsigned integer type (including
/// enum with an unsigned underlying type), or a typedef for such a type.
bool isUnsigned(mlir::Type Type);

//===--------------------------- Type categories --------------------------===//

/// Determine if the type is non-const. This is different from
/// `not Type.isConst()` in that the latter returns false for a typedef naming
/// a const-qualified type.
bool isModifiableType(mlir::Type Type);

/// Get the underlying primitive integer type of @p Type if it is either
/// * a primitive integer type, or
/// * an enum type, or
/// * a typedef naming any such type.
///
/// Otherwise null is returned. Qualifiers are ignored and the returned type is
/// always unqualified.
IntegerType getUnderlyingIntegerType(mlir::Type Type);

/// Determine if the specified type is a complete type. Only class types and
/// scalar tuple types can ever be incomplete. It is expected that types remain
/// incomplete only temporarily during construction of recursive types.
bool isCompleteType(mlir::Type Type);

/// Determine if the type is a scalar type, meaning either
/// * a primitive object type, or
/// * an enum type, or
/// * a pointer type, or
/// * a typedef naming any such type.
///
/// Qualifiers are ignored.
bool isScalarType(mlir::Type Type);

/// Determine if the type is a callable type, meaning either
/// * a function type, or
/// * a pointer-to-function type, or
/// * a typedef naming any such type.
///
/// Qualifiers are ignored.
bool isCallableType(mlir::Type Type);

/// If the type, after unwrapping typedefs, is a function type or a pointer to a
/// function type, returns that function type.
FunctionType getFunctionOrFunctionPointerFunctionType(mlir::Type Type);

} // namespace clift
