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
#define GEN_PASS_DEF_CLIFTBRANCHEQUALIZATION
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

static mlir::LogicalResult equalizeBranch(clift::BranchOpInterface Branch) {
  mlir::Region *FallthroughRegion = nullptr;

  for (mlir::Region &R : Branch.getBranchRegions()) {
    if (clift::getTrailingJumpOp(R))
      continue;

    if (FallthroughRegion)
      return mlir::failure();

    FallthroughRegion = &R;
  }

  if (not FallthroughRegion)
    return mlir::failure();

  if (FallthroughRegion->empty())
    FallthroughRegion->emplaceBlock();

  mlir::Block *Outer = Branch->getBlock();
  mlir::Block *Inner = &FallthroughRegion->front();

  mlir::Block::iterator Beg = std::next(Branch->getIterator());
  mlir::Block::iterator End = Outer->end();

  while (Beg != End and mlir::isa<clift::AssignLabelOp>(&*std::prev(End)))
    --End;

  Inner->getOperations().splice(Inner->end(),
                                Outer->getOperations(),
                                Beg,
                                End);

  return mlir::success();
}

struct BranchEqualizationPattern
  : mlir::OpInterfaceRewritePattern<clift::BranchOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::BranchOpInterface Branch,
                  mlir::PatternRewriter &Rewriter) const override {
    return equalizeBranch(Branch);
  }
};

struct BranchEqualizationPass
  : clift::impl::CliftBranchEqualizationBase<BranchEqualizationPass> {

  void runOnOperation() override {
#if 0
    mlir::RewritePatternSet Patterns(&getContext());
    Patterns.add<BranchEqualizationPattern>(Patterns.getContext());

    // TODO: Use walkAndApplyPatterns
    if (mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(Patterns))
        .failed())
      signalPassFailure();
#endif

    getOperation()->walk<mlir::WalkOrder::PreOrder>([](mlir::Operation *Op) {
      for (mlir::Region &R : Op->getRegions()) {
        for (auto Branch : R.getOps<clift::BranchOpInterface>())
          (void)equalizeBranch(Branch);
      }
    });
  }
};

} // namespace

clift::PassPtr<clift::FunctionOp> clift::createBranchEqualizationPass() {
  return std::make_unique<BranchEqualizationPass>();
}
