#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng-c/mlir/Dialect/Clift/IR/CliftAttributes.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftTypes.h"

namespace mlir::clift {

inline mlir::Type dealias(ValueType Type) {
  const auto GetTypedefTypeAttr = [](ValueType Type) -> TypedefTypeAttr {
    if (auto D = mlir::dyn_cast<DefinedType>(Type))
      return mlir::dyn_cast<TypedefTypeAttr>(D.getElementType());
    return nullptr;
  };

  while (auto Attr = GetTypedefTypeAttr(Type))
    Type = Attr.getUnderlyingType();

  return Type;
}

inline bool isVoid(ValueType Type) {
  Type = dealias(Type);

  if (auto T = mlir::dyn_cast<PrimitiveType>(Type))
    return T.getKind() == PrimitiveKind::VoidKind;

  return false;
}

inline bool isCompleteType(ValueType Type) {
  Type = dealias(Type);

  if (auto T = mlir::dyn_cast<DefinedType>(Type)) {
    auto Definition = T.getElementType();
    if (auto D = mlir::dyn_cast<StructTypeAttr>(Definition))
      return D.isDefinition();
    if (auto D = mlir::dyn_cast<UnionTypeAttr>(Definition))
      return D.isDefinition();
    return true;
  }

  if (auto T = mlir::dyn_cast<ScalarTupleType>(Type))
    return T.isComplete();

  if (auto T = mlir::dyn_cast<ArrayType>(Type))
    return isCompleteType(T.getElementType());

  return true;
}

inline bool isScalarType(ValueType Type) {
  Type = dealias(Type);

  if (auto T = mlir::dyn_cast<PrimitiveType>(Type))
    return T.getKind() != PrimitiveKind::VoidKind;

  if (auto T = mlir::dyn_cast<DefinedType>(Type))
    return mlir::isa<EnumTypeAttr>(T.getElementType());

  return mlir::isa<PointerType>(Type);
}

inline bool isObjectType(ValueType Type) {
  Type = dealias(Type);

  if (auto T = mlir::dyn_cast<PrimitiveType>(Type)) {
    if (T.getKind() == PrimitiveKind::VoidKind)
      return false;
  }

  if (auto T = mlir::dyn_cast<DefinedType>(Type)) {
    if (mlir::isa<FunctionTypeAttr>(T.getElementType()))
      return false;
  }

  if (mlir::isa<ScalarTupleType>(Type))
    return false;

  return true;
}

inline bool isArrayType(ValueType Type) {
  return mlir::isa<ArrayType>(dealias(Type));
}

} // namespace mlir::clift
