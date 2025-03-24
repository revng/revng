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

#include "revng/Support/Assert.h"
#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng/mlir/Dialect/Clift/IR/CliftInterfaces.h"

// This include should stay here for correct build procedure
#define GET_TYPEDEF_CLASSES
#include "revng/mlir/Dialect/Clift/IR/CliftOpsTypes.h.inc"

namespace mlir::clift {

struct TypedefDecomposition {
  ValueType Type;
  bool HasConstTypedef;
};

/// Recursively decomposes a typedef into its underlying non-typedef type and a
/// boolean indicating whether any of the typedefs added const. Note that the
/// underlying type itself may also be const while the boolean may be false.
TypedefDecomposition decomposeTypedef(ValueType Type);

/// Recursively remove typedefs and return the underlying type. If
/// @p IgnoreQualifiers is true, any qualifiers added by a typedef are applied
/// to the returned type. Otherwise such qualifiers are ignored.
ValueType dealias(ValueType Type, bool IgnoreQualifiers = false);

/// Remove top-level qualification from the given type, if it is a Clift value
/// type. Otherwise returns the type unchanged.
mlir::Type removeConst(mlir::Type Type);

/// Remove top-level qualification from the given type, if it is a Clift value
/// type. Otherwise returns the type unchanged.
template<typename TypeT>
TypeT removeConst(TypeT Type) {
  return mlir::cast<TypeT>(removeConst(static_cast<mlir::Type>(Type)));
}

/// Determine if the two types are equivalent, ignoring Clift qualifiers.
bool equivalent(mlir::Type Lhs, mlir::Type Rhs);

/// Determine if the type is non-const. This is different from
/// `not Type.isConst()` in that the latter returns false for a typedef naming
/// a const-qualified type.
bool isModifiableType(ValueType Type);

/// Determine if the specified primitive kind represents an integer type.
bool isIntegerKind(PrimitiveKind Kind);

/// Get the underlying primitive integer type of @p Type if it is either
/// * a primitive integer type, or
/// * an enum type, or
/// * a typedef naming any such type.
///
/// Otherwise null is returned. Qualifiers are ignored and the returned type is
/// always unqualified.
PrimitiveType getUnderlyingIntegerType(ValueType Type);

/// Determine if the specified type is a complete type. Only class types and
/// scalar tuple types can ever be incomplete. It is expected that types remain
/// incomplete only temporarily during construction of recursive types.
bool isCompleteType(ValueType Type);

/// Determine if the type is exactly void, ignoring type qualifiers.
bool isVoid(ValueType Type);

/// Determine if the type is a scalar type, meaning either
/// * a primitive object type, or
/// * an enum type, or
/// * a pointer type, or
/// * a typedef naming any such type.
///
/// Qualifiers are ignored.
bool isScalarType(ValueType Type);

/// Determine if the type is a primitive integer type, or a typedef naming such
/// a type, ignoring qualifiers.
bool isPrimitiveIntegerType(ValueType Type);

/// Determine if the type is an integer type. @see getUnderlyingIntegerType for
/// a breakdown of the set of integer types.
bool isIntegerType(ValueType Type);

/// Determine if the type is a floating point type, or a typedef naming such a
/// type, ignoring qualifiers.
bool isFloatType(ValueType Type);

/// Determine if the type is a pointer type. This includes pointers to objects
/// as well as pointers to functions. Qualifiers are ignored.
bool isPointerType(ValueType Type);

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
bool isObjectType(ValueType Type);

/// Determine if the type is an array type, unwrapping typedefs and ignoring
/// qualifiers.
bool isArrayType(ValueType Type);

/// Determine if the type is an enum type, unwrapping typedefs and ignoring
/// qualifiers
bool isEnumType(ValueType Type);

/// Determine if the type is a class type, meaning either a struct or union.
/// Qualifiers are ignored.
bool isClassType(ValueType Type);

/// Determine if the type is a function type, unwrapping typedefs and ignoring
/// qualifiers.
bool isFunctionType(ValueType Type);

/// Determine if the type is a callable type, meaning either
/// * a function type, or
/// * a pointer-to-function type, or
/// * a typedef naming any such type.
///
/// Qualifiers are ignored.
bool isCallableType(ValueType Type);

/// Determine if the type can be from a function. This means either
/// * void, or
/// * any object type except array types, or
/// * a scalar tuple type, or
/// * a typedef naming any such type.
///
/// Qualifiers are ignored.
bool isReturnableType(ValueType Type);

/// Get the underlying non-typedef type definition attribute, if any.
TypeDefinitionAttr getTypeDefinitionAttr(mlir::Type Type);

/// Get the underlying non-typedef type definition attribute of the specified
/// type, if any.
template<typename AttrT>
AttrT getTypeDefinitionAttr(mlir::Type Type) {
  return mlir::dyn_cast_or_null<AttrT>(getTypeDefinitionAttr(Type));
}

/// Get the underlying function type definition attribute, if any.
inline FunctionTypeAttr getFunctionTypeAttr(mlir::Type Type) {
  return getTypeDefinitionAttr<FunctionTypeAttr>(Type);
}

/// If the type, after unwrapping typedefs, is a function type or a pointer to a
/// function type, returns the underlying function definition attribute of that
/// function type.
FunctionTypeAttr getFunctionOrFunctionPointerTypeAttr(mlir::Type Type);

} // namespace mlir::clift
