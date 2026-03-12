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

namespace mlir::clift {

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
[[nodiscard]] inline ValueType unwrapTypedefs(ValueType Type) {
  return mlir::cast<ValueType>(unwrapTypedefs(static_cast<mlir::Type>(Type)));
}

/// Recursively unwraps typedefs and returns the underlying non-typedef type,
/// with any qualifiers from wrapping typedefs added onto the resulting type.
[[nodiscard]] mlir::Type collapseTypedefs(mlir::Type Type);

/// Recursively unwraps typedefs and returns the underlying non-typedef type,
/// with any qualifiers from wrapping typedefs added onto the resulting type.
[[nodiscard]] inline ValueType collapseTypedefs(ValueType Type) {
  return mlir::cast<ValueType>(collapseTypedefs(static_cast<mlir::Type>(Type)));
}

//===---------------------------- CV-Qualifiers ---------------------------===//

/// Determine if the type is top-level const qualified.
[[nodiscard]] bool isConst(mlir::Type Type);

/// Add top-level qualification to the given type, if it is a ValueType.
/// Otherwise returns the type unchanged.
[[nodiscard]] mlir::Type addConst(mlir::Type Type);

/// Add top-level qualification to the given type, if it is a ValueType.
/// Otherwise returns the type unchanged.
template<typename TypeT>
[[nodiscard]] TypeT addConst(TypeT Type) {
  return mlir::cast<TypeT>(addConst(mlir::Type(Type)));
}

/// Remove top-level qualification from the given type, if it is a ValueType.
/// Otherwise returns the type unchanged.
[[nodiscard]] mlir::Type removeConst(mlir::Type Type);

/// Remove top-level qualification from the given type, if it is a ValueType.
/// Otherwise returns the type unchanged.
template<typename TypeT>
[[nodiscard]] TypeT removeConst(TypeT Type) {
  return mlir::cast<TypeT>(removeConst(mlir::Type(Type)));
}

/// Determine if the two types are equivalent, ignoring Clift CV-qualifiers.
[[nodiscard]] bool equivalent(mlir::Type Lhs, mlir::Type Rhs);

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

/// Determine if the type is exactly void, ignoring type qualifiers.
bool isVoid(mlir::Type Type);

/// Determine if the type is a scalar type, meaning either
/// * a primitive object type, or
/// * an enum type, or
/// * a pointer type, or
/// * a typedef naming any such type.
///
/// Qualifiers are ignored.
bool isScalarType(mlir::Type Type);

IntegerType getPrimitiveIntegerType(mlir::Type Type);

/// Determine if the type is a primitive integer type, or a typedef naming such
/// a type, ignoring qualifiers.
bool isPrimitiveIntegerType(mlir::Type Type);

/// Determine if the type is an integer type. @see getUnderlyingIntegerType for
/// a breakdown of the set of integer types.
bool isIntegerType(mlir::Type Type);

/// Determine if the type is a "boolean" type, i.e. a signed primitive integer
/// type.
bool isBooleanType(mlir::Type Type);

/// Determine if the type is a floating point type, or a typedef naming such a
/// type, ignoring qualifiers.
bool isFloatType(mlir::Type Type);

PointerType getPointerType(mlir::Type Type);

/// Determine if the type is a pointer type. This includes pointers to objects
/// as well as pointers to functions. Qualifiers are ignored.
bool isPointerType(mlir::Type Type);

/// Determine if the type is an object type. This is the set of types
/// representing program objects. In other words it is the set of types which
/// can be used to declare a variable, meaning either
/// * a non-void primitive type, or
/// * a pointer type, or
/// * an array type, or
/// * an enum type, or
/// * a class type, or
/// * a typedef naming any such type.
///
/// Qualifiers are ignored.
bool isObjectType(mlir::Type Type);

/// Determine if the type is an array type, unwrapping typedefs and ignoring
/// qualifiers.
bool isArrayType(mlir::Type Type);

/// Determine if the type is an enum type, unwrapping typedefs and ignoring
/// qualifiers
bool isEnumType(mlir::Type Type);

/// Determine if the type is a class type, meaning either a struct or union.
/// Qualifiers are ignored.
bool isClassType(mlir::Type Type);

/// Determine if the type is a function type, unwrapping typedefs and ignoring
/// qualifiers.
bool isFunctionType(mlir::Type Type);

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

/// Verify that the type is a valid function return type, meaning:
/// * void, or
/// * any object type except an array type, or
/// * a typedef naming any such type, and
/// * is not a type composed of any non-zero number of pointer indirections to
///   an array or function (not involving any typedefs).
///
/// Qualifiers are ignored.
mlir::LogicalResult
verifyReturnType(llvm::function_ref<mlir::InFlightDiagnostic()> EmitError,
                 mlir::Type Type);

} // namespace mlir::clift
