#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>

#include "mlir/IR/PatternMatch.h"

#include "revng/Clift/CliftTypes.h"

namespace clift {

[[nodiscard]] std::partial_ordering compareTypeRefinement(mlir::Type T1,
                                                          mlir::Type T2);

void populateWithTypeRefinementPatterns(mlir::RewritePatternSet &Set);

} // namespace clift
