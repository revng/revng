//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <ranges>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/CliftTransforms/Expressions.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/CliftTransforms/RewriteHelpers.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTUNIQUEFALLTHROUGHBRANCHHOISTING
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

struct UniqueFallthroughBranchHoistingPattern
  : mlir::OpInterfaceRewritePattern<BranchOpInterface> {

  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(BranchOpInterface Branch,
                  mlir::PatternRewriter &Rewriter) const override {

    mlir::Region *FallthroughRegion = nullptr;

    // Find unique fallthrough region, if any.
    for (mlir::Region &R : Branch.getBranchRegions()) {
      if (indirectlyFallsThrough(R)) {
        if (FallthroughRegion != nullptr)
          return mlir::failure();

        FallthroughRegion = &R;
      }
    }

    if (FallthroughRegion == nullptr or FallthroughRegion->empty())
      return mlir::failure();

    hoistBranchRegion(Rewriter, *FallthroughRegion);
    return mlir::success();
  }
};

template<typename T>
using PassBase = clift::impl::CliftUniqueFallthroughBranchHoistingBase<T>;

struct UniqueFallthroughBranchHoistingPass
  : PassBase<UniqueFallthroughBranchHoistingPass> {

  mlir::FrozenRewritePatternSet Patterns;

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);
    Set.add<UniqueFallthroughBranchHoistingPattern>(Context);
    Patterns = mlir::FrozenRewritePatternSet(std::move(Set));
    return mlir::success();
  }

  void runOnOperation() override {
    // TODO: Use walkAndApplyPatterns
    if (mlir::applyPatternsAndFoldGreedily(getOperation(), Patterns).failed())
      signalPassFailure();
  }
};

} // namespace

PassPtr<FunctionOp> clift::createUniqueFallthroughBranchHoistingPass() {
  return std::make_unique<UniqueFallthroughBranchHoistingPass>();
}
