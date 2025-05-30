#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace mlir {

class RewritePatternSet;

namespace clift {

void populateBeautifyStatementRewritePatterns(RewritePatternSet &Patterns);
void populateBeautifyExpressionRewritePatterns(RewritePatternSet &Patterns);

} // namespace clift
} // namespace mlir
