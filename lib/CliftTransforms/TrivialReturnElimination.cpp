//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/CliftTransforms/RewriteHelpers.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTTRIVIALRETURNELIMINATION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

struct TrivialReturnEliminationPattern : mlir::OpRewritePattern<ReturnOp> {

  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ReturnOp Return,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not Return.getResult().empty())
      return mlir::failure();

    auto Function = Return->getParentOfType<FunctionOp>();
    auto JumpTarget = BlockPosition::getEnd(Function.getBody());
    auto FallTarget = getFallthroughTarget(BlockPosition::getNext(Return));

    if (JumpTarget != FallTarget)
      return mlir::failure();

    Rewriter.eraseOp(Return);
    return mlir::success();
  }
};

template<typename T>
using PassBase = clift::impl::CliftTrivialReturnEliminationBase<T>;

struct TrivialReturnEliminationPass : PassBase<TrivialReturnEliminationPass> {
  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();

    mlir::RewritePatternSet Patterns(Context);
    Patterns.add<TrivialReturnEliminationPattern>(Context);

    // TODO: Use walkAndApplyPatterns
    if (mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(Patterns))
          .failed())
      signalPassFailure();
  }
};

} // namespace

PassPtr<FunctionOp> clift::createTrivialReturnEliminationPass() {
  return std::make_unique<TrivialReturnEliminationPass>();
}
