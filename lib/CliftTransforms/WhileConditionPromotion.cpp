//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/CliftTransforms/RewriteHelpers.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTWHILECONDITIONPROMOTION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

/// Converts while (true) statements with leading conditional breaks into
/// conditional while statements.
struct WhileConditionPromotionPattern : mlir::OpRewritePattern<WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(WhileOp While,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not isTriviallyTrue(While.getCondition()))
      return mlir::failure();

    // If the while-loop continue label has any users, converting to a
    // do-while-loop would change the meaning of any jumps targeting that label.
    if (auto Continue = While.getContinueLabel()) {
      if (not Continue.use_empty())
        return mlir::failure();
    }

    // Check for an if-statement at the beginning of the loop body:
    auto If = clift::getFirstOp<IfOp>(While.getBody());
    if (not If)
      return mlir::failure();

    auto IsBreak = [&While](mlir::Region &R) -> bool {
      auto Last = clift::getOnlyOp<BreakToOp>(R);
      return Last and Last.getLabelAssignmentOp() == While;
    };

    bool ThenBreak = IsBreak(If.getThen());
    bool ElseBreak = IsBreak(If.getElse());

    // If neither branch contains a break, this is not a conditional while.
    if (not ThenBreak and not ElseBreak)
      return mlir::failure();

    // The region opposite the break-region must be empty.
    if (not isEmptyRegionOrBlock(ThenBreak ? If.getElse() : If.getThen()))
      return mlir::failure();

    if (ThenBreak) {
      // With the break in the true branch, the condition must be inverted.
      invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());
    }

    // Remove the existing loop condition.
    While.getCondition().front().erase();

    // The if-statement condition is inlined into the while condition.
    inlineRegionAtEnd(Rewriter, If.getCondition(), While.getCondition());

    // Finally, the if-statement can be erased.
    Rewriter.eraseOp(If);

    return mlir::success();
  }
};

struct WhileConditionPromotionPass
  : impl::CliftWhileConditionPromotionBase<WhileConditionPromotionPass> {

  mlir::FrozenRewritePatternSet Patterns;

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);
    Set.add<WhileConditionPromotionPattern>(Set.getContext());
    Patterns = mlir::FrozenRewritePatternSet(std::move(Set));
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

PassPtr<FunctionOp> clift::createWhileConditionPromotionPass() {
  return std::make_unique<WhileConditionPromotionPass>();
}
