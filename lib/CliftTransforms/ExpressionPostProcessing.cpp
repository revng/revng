//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/ExpressionHelpers.h"
#include "revng/CliftTransforms/Expressions.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTEXPRESSIONPOSTPROCESSING
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {
namespace expression_post_processing {

static bool isNegativeImmediateValue(llvm::APInt Value) {
  return Value.isSignBitSet() and not Value.isMinSignedValue();
}

static llvm::APInt negateIntegerValue(llvm::APInt Value) {
  Value.negate();
  return Value;
}

#include "revng/CliftTransforms/ExpressionPostProcessing.h.inc"

} // namespace expression_post_processing

struct ExpressionPostProcessingPass
  : impl::CliftExpressionPostProcessingBase<ExpressionPostProcessingPass> {

  mlir::FrozenRewritePatternSet Patterns;

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);

    expression_post_processing::populateWithGenerated(Set);
    populateWithCastCanonicalizations(Set);

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

PassPtr<FunctionOp> clift::createExpressionPostProcessingPass() {
  return std::make_unique<ExpressionPostProcessingPass>();
}
