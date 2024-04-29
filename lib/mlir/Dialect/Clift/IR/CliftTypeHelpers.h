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

} // namespace mlir::clift
