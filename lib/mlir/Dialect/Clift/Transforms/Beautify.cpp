#if 0
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"
#include "revng/mlir/Dialect/Clift/Transforms/Rewrites.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTBEAUTIFY
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

struct BeautifyPass : clift::impl::CliftBeautifyBase<BeautifyPass> {
  void runOnOperation() override {
    mlir::MLIRContext &Context = getContext();

    mlir::RewritePatternSet Patterns(&Context);
    clift::populateBeautifyStatementRewritePatterns(Patterns);
    clift::populateBeautifyExpressionRewritePatterns(Patterns);

    if (mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(Patterns))
          .failed())
      signalPassFailure();
  }
};

} // namespace

clift::PassPtr<clift::FunctionOp> clift::createBeautifyPass() {
  return std::make_unique<BeautifyPass>();
}
#endif
