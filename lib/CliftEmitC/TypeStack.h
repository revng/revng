#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/CliftTypeInterfaces.h"

namespace mlir::clift {

struct TypeStackItem {
  enum class ItemKind {
    // Last item is always one of:
    Primitive,
    Defined,
    ForeignPointer,

    // All the other items can be:
    Pointer,
    Array,
  };

  ItemKind Kind;
  ValueType Type;
};

/// Recurse through the declaration, pushing each level onto the stack until
/// a terminal type is encountered.
///
/// The terminal types include:
/// - primitives,
/// - defined types,
/// - pointers with size not matching the target implementation pointer size.
llvm::SmallVector<TypeStackItem> makeTypeStack(ValueType Type,
                                               uint64_t TargetPointerSize);

} // namespace mlir::clift
