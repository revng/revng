#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"

namespace clift {

void populateWithBooleanNegationPatterns(mlir::RewritePatternSet &Set);

} // namespace clift
