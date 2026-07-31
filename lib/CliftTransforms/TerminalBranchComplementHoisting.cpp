//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/CliftTransforms/Expressions.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/CliftTransforms/RewriteHelpers.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTTERMINALBRANCHCOMPLEMENTHOISTING
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

enum class HoistingTarget : bool {
  Then,
  Else,
};

static bool hasSingleNoFallthrough(mlir::Region &R) {
  return static_cast<bool>(getLastNoFallthroughStatement(R));
}

// Computes an approximation of the size of a statement region in C.
static unsigned approximateRegionWeight(mlir::Region &R) {
  revng_assert(R.hasOneBlock());

  unsigned Weight = 0;
  R.walk([&Weight](mlir::Operation *Op) {
    // Label declarations can be ignored, as they have no C representation.
    if (mlir::isa<MakeLabelOp>(Op))
      return;

    // Neither expression statement nor yield operations have a direct C
    // representation.
    if (mlir::isa<ExpressionStatementOp, YieldOp>(Op))
      return;

    ++Weight;
  });
  return Weight;
}

// Attempts to select one branch of an if-statement to be inlined into the
// nesting scope. If neither branch should be hoisted, the result is nullopt.
static std::optional<HoistingTarget> selectHoistingTarget(IfOp If) {
  // Both empty region shapes are meaningful for an else, and are handled in
  // turn: a block-less region is an absent else clause, leaving nothing to
  // hoist, while an empty block is an else clause doing nothing, which is
  // dropped.
  if (If.getElse().empty())
    return std::nullopt;

  if (If.getElse().front().empty())
    return HoistingTarget::Else;

  // A then clause is instead always present, so its two empty region shapes
  // mean the same thing and are tested together. Inverting the condition drops
  // an empty then.
  if (isEmptyRegionOrBlock(If.getThen()))
    return HoistingTarget::Then;

  bool ThenFallthrough = indirectlyFallsThrough(If.getThen());
  bool ElseFallthrough = indirectlyFallsThrough(If.getElse());

  // If both branches fall through, neither can be hoisted.
  if (ThenFallthrough and ElseFallthrough)
    return std::nullopt;

  // If exactly one branch falls through, that one is hoisted.
  if (ThenFallthrough != ElseFallthrough)
    return static_cast<HoistingTarget>(ElseFallthrough);

  bool ThenHasDirectNoFallthrough = hasSingleNoFallthrough(If.getThen());
  bool ElseHasDirectNoFallthrough = hasSingleNoFallthrough(If.getElse());

  // If exactly one branch contains a single directly non-fallthrough operation,
  // that one is hoisted.
  if (ThenHasDirectNoFallthrough != ElseHasDirectNoFallthrough)
    return static_cast<HoistingTarget>(ElseHasDirectNoFallthrough);

  // Otherwise, the sizes of both branches are approximated, and a decision is
  // made by comparing those approximations.
  unsigned ThenWeight = approximateRegionWeight(If.getThen());
  unsigned ElseWeight = approximateRegionWeight(If.getElse());

  // Hoist the region with the lower weight (or else if equal).
  return static_cast<HoistingTarget>(ElseWeight <= ThenWeight);
}

struct TerminalBranchComplementHoistingPattern
  : mlir::OpRewritePattern<clift::IfOp> {

  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::IfOp If,
                  mlir::PatternRewriter &Rewriter) const override {
    auto OptTarget = selectHoistingTarget(If);
    if (not OptTarget)
      return mlir::failure();
    auto Target = *OptTarget;

    if (Target == HoistingTarget::Then)
      invertIfStatement(Rewriter, If);

    if (mlir::Block *ElseBlock = getOnlyBlock(If.getElse())) {
      Rewriter.updateRootInPlace(If.getOperation(), [&]() {
        inlineBlockBefore(Rewriter,
                          ElseBlock,
                          If->getBlock(),
                          std::next(If->getIterator()));
      });

      Rewriter.eraseBlock(ElseBlock);
    }

    return mlir::success();
  }
};

template<typename T>
using PassBase = clift::impl::CliftTerminalBranchComplementHoistingBase<T>;

struct TerminalBranchComplementHoistingPass
  : PassBase<TerminalBranchComplementHoistingPass> {

  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();
    FunctionOp Function = getOperation();

    // Apply terminal branch complement hoisting:
    {
      mlir::RewritePatternSet Patterns(Context);
      Patterns.add<TerminalBranchComplementHoistingPattern>(Context);

      // TODO: Use walkAndApplyPatterns
      if (mlir::applyPatternsAndFoldGreedily(Function, std::move(Patterns))
            .failed())
        signalPassFailure();
    }

    // Terminal branch complement hoisting may need to invert if-statements.
    // That introduces negated conditions, e.g. `!!x`. This rewrite undoes them:
    {
      mlir::RewritePatternSet Patterns(Context);
      populateWithBooleanNegationPatterns(Patterns);

      // TODO: Use walkAndApplyPatterns
      if (mlir::applyPatternsAndFoldGreedily(Function, std::move(Patterns))
            .failed())
        signalPassFailure();
    }
  }
};

} // namespace

PassPtr<FunctionOp> clift::createTerminalBranchComplementHoistingPass() {
  return std::make_unique<TerminalBranchComplementHoistingPass>();
}
