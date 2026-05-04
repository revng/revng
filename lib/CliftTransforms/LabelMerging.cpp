//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTLABELMERGING
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

struct AssignLabelMergingPattern : mlir::OpRewritePattern<AssignLabelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AssignLabelOp AssignLabel,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Block::iterator Pos = std::next(AssignLabel->getIterator());

    if (Pos == AssignLabel->getBlock()->end())
      return mlir::failure();

    auto NextAssignLabel = mlir::dyn_cast<AssignLabelOp>(&*Pos);
    if (not NextAssignLabel)
      return mlir::failure();

    Rewriter.replaceAllUsesWith(NextAssignLabel.getLabel(),
                                AssignLabel.getLabel());

    Rewriter.eraseOp(NextAssignLabel.getOperation());

    return mlir::success();
  }
};

struct LoopLabelMergingPattern
  : mlir::OpInterfaceRewritePattern<LoopOpInterface> {

  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LoopOpInterface Loop,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::LogicalResult Result = mlir::failure();

    auto ReplaceLabel = [&Rewriter, &Result](AssignLabelOp Assignment,
                                             mlir::Value NewLabel) {
      Rewriter.replaceAllUsesWith(Assignment.getLabel(), NewLabel);
      Rewriter.eraseOp(Assignment);
      Result = mlir::success();
    };

    if (mlir::Value BreakLabel = Loop.getBreakLabel()) {
      if (auto Assignment = getNextOp<AssignLabelOp>(Loop))
        ReplaceLabel(Assignment, BreakLabel);
    }

    if (mlir::Value ContinueLabel = Loop.getContinueLabel()) {
      if (auto Assignment = getLastOp<AssignLabelOp>(Loop.getBody()))
        ReplaceLabel(Assignment, ContinueLabel);
    }

    return Result;
  }
};

struct LabelMergingPass : clift::impl::CliftLabelMergingBase<LabelMergingPass> {

  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();

    mlir::RewritePatternSet Patterns(Context);
    Patterns.add(MakeLabelOp::canonicalize);
    Patterns.add<AssignLabelMergingPattern>(Context);
    Patterns.add<LoopLabelMergingPattern>(Context);

    // TODO: Use walkAndApplyPatterns
    if (mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(Patterns))
          .failed())
      signalPassFailure();
  }
};

} // namespace

PassPtr<FunctionOp> clift::createLabelMergingPass() {
  return std::make_unique<LabelMergingPass>();
}
