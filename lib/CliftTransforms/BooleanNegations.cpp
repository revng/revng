//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Expressions.h"

using namespace clift;

namespace {

#include "revng/CliftTransforms/BooleanNegations.h.inc"

} // namespace

void clift::populateWithBooleanNegationPatterns(mlir::RewritePatternSet &Set) {
  populateWithGenerated(Set);
}
