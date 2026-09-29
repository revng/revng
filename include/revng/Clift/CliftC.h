#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"

namespace clift {

/// Returns true if the two types are equivalent in C. Typedefs are not
/// considered equivalent, regardless of their underlying types. Note that
/// unlike clift::equivalent, this function does not ignore qualifiers.
[[nodiscard]] bool equivalentInC(mlir::Type LHS, mlir::Type RHS);

/// Returns true if the value represents a null pointer constant in C.
[[nodiscard]] bool isNullPointerConstantInC(mlir::Value Value);

/// Returns true if the conversion, in C, from the source type to the target
/// type is implicit.
[[nodiscard]] bool isImplicitlyConvertibleInC(mlir::Type Source,
                                              mlir::Type Target,
                                              const CDialect &Dialect);

/// Returns true if the conversion, in C, represented by the cast operation is
/// implicit, taking into account the semantics of the converted operand.
[[nodiscard]] bool isImplicitConversionInC(CastOpInterface Cast,
                                           const CDialect &Dialect);

} // namespace clift
