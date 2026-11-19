//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>

#include "llvm/ADT/SmallPtrSet.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/CliftTransforms/RewriteHelpers.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTVOIDRETURNTOGOTOCONVERSION
#define GEN_PASS_DEF_CLIFTJUMPTOVOIDRETURNCONVERSION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

using ValueSet = llvm::SmallPtrSet<mlir::Value, 1>;

struct VoidReturnToGotoConversionPattern : mlir::OpRewritePattern<ReturnOp> {
  mlir::Value Label;

  explicit VoidReturnToGotoConversionPattern(mlir::MLIRContext *Context,
                                             mlir::Value Label) :
    OpRewritePattern(Context), Label(Label) {}

  mlir::LogicalResult
  matchAndRewrite(ReturnOp Return,
                  mlir::PatternRewriter &Rewriter) const override {
    auto Function = Return->getParentOfType<FunctionOp>();
    revng_assert(clift::unwrapped_isa<VoidType>(Function.getReturnType()));

    Rewriter.setInsertionPointAfter(Return);
    Rewriter.create<GotoOp>(Return.getLoc(), Label);
    Rewriter.eraseOp(Return);

    return mlir::success();
  }
};

struct JumpToVoidReturnConversionPattern
  : mlir::OpInterfaceRewritePattern<JumpStatementOpInterface> {

  BlockPosition ReturnPosition;

  explicit JumpToVoidReturnConversionPattern(mlir::MLIRContext *Context,
                                             FunctionOp Function) :
    OpInterfaceRewritePattern(Context),
    ReturnPosition(BlockPosition::getEnd(Function.getBody())) {}

  mlir::LogicalResult
  matchAndRewrite(JumpStatementOpInterface Jump,
                  mlir::PatternRewriter &Rewriter) const override {

    if (getFallthroughTarget(getJumpTarget(Jump)) != ReturnPosition)
      return mlir::failure();

    Rewriter.setInsertionPointAfter(Jump);
    Rewriter.create<ReturnOp>(Jump->getLoc());
    Rewriter.eraseOp(Jump);

    return mlir::success();
  }
};

struct VoidReturnToGotoConversionPass
  : impl::CliftVoidReturnToGotoConversionBase<VoidReturnToGotoConversionPass> {

  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();

    FunctionOp Function = getOperation();
    if (Function.isExternal())
      return;

    if (not clift::unwrapped_isa<VoidType>(Function.getReturnType()))
      return;

    mlir::Location Loc = mlir::UnknownLoc::get(Context);

    mlir::OpBuilder Builder(Function.getBody());
    mlir::Value Label = Builder.create<MakeLabelOp>(Loc);

    Builder.setInsertionPointToEnd(Builder.getBlock());
    Builder.create<AssignLabelOp>(Loc, Label);

    mlir::RewritePatternSet Patterns(Context);
    Patterns.add<VoidReturnToGotoConversionPattern>(Context, Label);

    // TODO: Use walkAndApplyPatterns
    if (mlir::applyPatternsAndFoldGreedily(Function, std::move(Patterns))
          .failed())
      signalPassFailure();
  }
};

struct JumpToVoidReturnConversionPass
  : impl::CliftJumpToVoidReturnConversionBase<JumpToVoidReturnConversionPass> {

  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();

    FunctionOp Function = getOperation();
    if (Function.isExternal())
      return;

    if (not clift::unwrapped_isa<VoidType>(Function.getReturnType()))
      return;

    mlir::RewritePatternSet Patterns(Context);
    Patterns.add<JumpToVoidReturnConversionPattern>(Context, Function);
    Patterns.add(MakeLabelOp::canonicalize);

    // TODO: Use walkAndApplyPatterns
    if (mlir::applyPatternsAndFoldGreedily(Function, std::move(Patterns))
          .failed())
      signalPassFailure();
  }
};

} // namespace

PassPtr<FunctionOp> clift::createVoidReturnToGotoConversionPass() {
  return std::make_unique<VoidReturnToGotoConversionPass>();
}

PassPtr<FunctionOp> clift::createJumpToVoidReturnConversionPass() {
  return std::make_unique<JumpToVoidReturnConversionPass>();
}
