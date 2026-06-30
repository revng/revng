#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"

namespace clift {

void populateWithCastCanonicalizations(mlir::RewritePatternSet &Set);
void populateWithBooleanNegationPatterns(mlir::RewritePatternSet &Set);
void populateWithExpressionOptimizationPatterns(mlir::RewritePatternSet &Set);

} // namespace clift
