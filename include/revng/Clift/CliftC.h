#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"

namespace clift::c {

/// Returns true if the conversion, in C, from the source type to the target
/// type is implicit.
[[nodiscard]] bool isImplicitlyConvertible(mlir::Type Source,
                                           mlir::Type Target);

/// Returns true if the conversion, in C, represented by the cast operation is
/// implicit, taking into account the semantics of the converted operand.
[[nodiscard]] bool isImplicitConversion(CastOpInterface Cast);

} // namespace clift::c
