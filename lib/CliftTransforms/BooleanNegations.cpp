//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Expressions.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTBOOLEANNEGATIONSIMPLIFICATION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

namespace boolean_negations {
#include "revng/CliftTransforms/BooleanNegations.h.inc"
} // namespace boolean_negations

template<typename T>
using PassBase = impl::CliftBooleanNegationSimplificationBase<T>;

struct BooleanNegationSimplificationPass
  : PassBase<BooleanNegationSimplificationPass> {

  mlir::FrozenRewritePatternSet Patterns;

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);

    populateWithBooleanNegationPatterns(Set);
    Patterns = mlir::FrozenRewritePatternSet(std::move(Set),
                                             disabledPatterns,
                                             enabledPatterns);

    return mlir::success();
  }

  void runOnOperation() override {
    FunctionOp Function = getOperation();
    mlir::Region &Body = Function.getBody();

    if (Body.empty())
      return;

    mlir::GreedyRewriteConfig Config;
    Config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
    if (mlir::applyPatternsAndFoldGreedily(Function, Patterns, Config).failed())
      signalPassFailure();
  }
};

} // namespace

void clift::populateWithBooleanNegationPatterns(mlir::RewritePatternSet &Set) {
  boolean_negations::populateWithGenerated(Set);
}

PassPtr<FunctionOp> clift::createBooleanNegationSimplificationPass() {
  return std::make_unique<BooleanNegationSimplificationPass>();
}
