//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/mlir/Dialect/Clift/IR/CliftOpHelpers.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTRETURNINTOGOTOCONVERSION
#define GEN_PASS_DEF_CLIFTGOTOINTORETURNCONVERSION
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

struct ReturnIntoGotoConversionPattern
  : mlir::OpRewritePattern<clift::ReturnOp> {
  mlir::Value Label;

  explicit ReturnIntoGotoConversionPattern(mlir::MLIRContext *Context,
                                           mlir::Value Label) :
    OpRewritePattern(Context), Label(Label) {}

  mlir::LogicalResult
  matchAndRewrite(clift::ReturnOp Return,
                  mlir::PatternRewriter &Rewriter) const override {
    auto Function = Return->getParentOfType<clift::FunctionOp>();
    revng_assert(clift::isVoid(Function.getCliftReturnType()));

    Rewriter.setInsertionPointAfter(Return);
    Rewriter.create<clift::GoToOp>(Return.getLoc(), Label);
    Rewriter.eraseOp(Return);

    return mlir::success();
  }
};

struct ReturnIntoGotoConversionPass
  : clift::impl::CliftReturnIntoGotoConversionBase<
      ReturnIntoGotoConversionPass> {

  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();

    clift::FunctionOp Function = getOperation();
    if (not clift::isVoid(Function.getCliftReturnType()))
      return;

    mlir::Location Loc = mlir::UnknownLoc::get(Context);

    mlir::OpBuilder Builder(Function.getBody());
    mlir::Value Label = Builder.create<clift::MakeLabelOp>(Loc);

    Builder.setInsertionPointToEnd(Builder.getBlock());
    Builder.create<clift::AssignLabelOp>(Loc, Label);

    mlir::RewritePatternSet Patterns(Context);
    Patterns.add<ReturnIntoGotoConversionPattern>(Context, Label);

    // TODO: Use walkAndApplyPatterns
    if (mlir::applyPatternsAndFoldGreedily(Function, std::move(Patterns))
          .failed())
      signalPassFailure();
  }
};

struct GotoIntoReturnConversionPattern : mlir::OpRewritePattern<clift::GoToOp> {
  mlir::Value Label;

  explicit GotoIntoReturnConversionPattern(mlir::MLIRContext *Context,
                                           mlir::Value Label) :
    OpRewritePattern(Context), Label(Label) {}

  mlir::LogicalResult
  matchAndRewrite(clift::GoToOp Goto,
                  mlir::PatternRewriter &Rewriter) const override {
    auto Function = Goto->getParentOfType<clift::FunctionOp>();
    revng_assert(clift::isVoid(Function.getCliftReturnType()));

    if (Goto.getLabel() != Label)
      return mlir::failure();

    Rewriter.setInsertionPointAfter(Goto);
    Rewriter.create<clift::ReturnOp>(Goto.getLoc());
    Rewriter.eraseOp(Goto);

    return mlir::success();
  }
};

struct GotoIntoReturnConversionPass
  : clift::impl::CliftGotoIntoReturnConversionBase<
      GotoIntoReturnConversionPass> {

  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();

    clift::FunctionOp Function = getOperation();
    if (not clift::isVoid(Function.getCliftReturnType()))
      return;

    auto Label = clift::getTrailingOp<clift::AssignLabelOp>(Function.getBody());
    if (not Label)
      return;

    mlir::RewritePatternSet Patterns(Context);
    Patterns.add<GotoIntoReturnConversionPattern>(Context, Label.getLabel());
    Patterns.add(clift::MakeLabelOp::canonicalize);

    // TODO: Use walkAndApplyPatterns
    if (mlir::applyPatternsAndFoldGreedily(Function, std::move(Patterns))
          .failed())
      signalPassFailure();
  }
};

} // namespace

clift::PassPtr<clift::FunctionOp> clift::createReturnIntoGotoConversionPass() {
  return std::make_unique<ReturnIntoGotoConversionPass>();
}

clift::PassPtr<clift::FunctionOp> clift::createGotoIntoReturnConversionPass() {
  return std::make_unique<GotoIntoReturnConversionPass>();
}
