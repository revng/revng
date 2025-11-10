//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/CliftTypes.h"

#include "TypeStack.h"

namespace mlir::clift {

llvm::SmallVector<TypeStackItem> makeTypeStack(ValueType Type,
                                               uint64_t TargetPointerSize) {
  llvm::SmallVector<TypeStackItem> Result;

  while (true) {
    if (mlir::isa<mlir::clift::PrimitiveType>(Type)) {
      Result.emplace_back(TypeStackItem::ItemKind::Primitive, Type);
      break;

    } else if (auto Pointer = mlir::dyn_cast<mlir::clift::PointerType>(Type)) {
      if (Pointer.getPointerSize() == TargetPointerSize) {
        Result.emplace_back(TypeStackItem::ItemKind::Pointer, Pointer);
        Type = Pointer.getPointeeType();

      } else {
        Result.emplace_back(TypeStackItem::ItemKind::ForeignPointer, Pointer);
        break;
      }

    } else if (auto Array = mlir::dyn_cast<mlir::clift::ArrayType>(Type)) {
      Result.emplace_back(TypeStackItem::ItemKind::Array, Array);
      Type = Array.getElementType();

    } else if (mlir::isa<mlir::clift::DefinedType>(Type)) {
      Result.emplace_back(TypeStackItem::ItemKind::Defined, Type);
      break;

    } else {
      revng_abort("Unsupported Type kind.");
    }
  }

  return Result;
}

} // namespace mlir::clift
