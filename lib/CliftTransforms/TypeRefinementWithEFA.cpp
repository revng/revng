//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/CliftTransforms/EmitFieldAccesses.h"
#include "revng/CliftTransforms/Expressions.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/CliftTransforms/TypeRefinement.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTTYPEREFINEMENTWITHEFA
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

struct TypeRefinementWithEFAPass
  : clift::impl::CliftTypeRefinementWithEFABase<TypeRefinementWithEFAPass> {

  mlir::FrozenRewritePatternSet Patterns;

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);

    populateWithEmitFieldAccessesPatterns(Set);
    populateWithTypeRefinementPatterns(Set);
    populateWithExpressionOptimizationPatterns(Set);

    Patterns = mlir::FrozenRewritePatternSet(std::move(Set),
                                             disabledPatterns,
                                             enabledPatterns);

    return mlir::success();
  }

  void runOnOperation() override {
    [[maybe_unused]] EFAThreadCache EFACache;

    mlir::GreedyRewriteConfig Config;
    Config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
    if (mlir::applyPatternsAndFoldGreedily(getOperation(), Patterns, Config)
          .failed())
      signalPassFailure();
  }
};

} // namespace

PassPtr<clift::FunctionOp> clift::createTypeRefinementWithEFAPass() {
  return std::make_unique<TypeRefinementWithEFAPass>();
}
