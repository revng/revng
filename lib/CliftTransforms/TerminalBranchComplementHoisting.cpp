//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <algorithm>
#include <ranges>

#include "llvm/ADT/SmallVector.h"

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

// Computes an approximation of the size of a statement region in C.
static unsigned approximateRegionWeight(mlir::Region &R) {
  if (R.empty())
    return 0;

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

// A non-fallthrough region leaves in a definite way when control does so in a
// single known manner - a continue, break, goto or return - as opposed to a
// Mixed region, whose nested branches leave in differing ways.
static bool leavesDefinitely(mlir::Region &R) {
  return isIndirectlyNoFallthrough(R) != NoFallthroughKind::Mixed;
}

// When no branch region falls through, any of them may be hoisted. This
// generalises the heuristic originally used to choose between the two branches
// of an if: prefer a region leaving in a definite way, then the lowest weight.
static unsigned selectByHeuristic(llvm::MutableArrayRef<mlir::Region> Regions) {
  auto Rank = [&Regions](unsigned I) {
    return std::pair(not leavesDefinitely(Regions[I]),
                     approximateRegionWeight(Regions[I]));
  };

  // Ranking the regions back to front resolves a tie towards the later one,
  // since the minimum of several equivalent elements is the first of them.
  auto Indices = std::views::iota(0u, static_cast<unsigned>(Regions.size()))
                 | std::views::reverse;

  return std::ranges::min(Indices, std::less<>(), Rank);
}

// Selects the branch region to be inlined into the nesting scope, as an index
// into \p Regions. If none should be hoisted, the result is nullopt.
//
// The choice is deliberately expressed on the branch regions alone, so that
// every branch statement makes it in this single place.
static std::optional<unsigned>
selectHoistingTarget(llvm::MutableArrayRef<mlir::Region> Regions) {
  llvm::SmallVector<unsigned> Fallthrough;
  for (unsigned I = 0; I < Regions.size(); ++I) {
    if (indirectlyFallsThrough(Regions[I]))
      Fallthrough.push_back(I);
  }

  // With two or more fall-through branches, hoisting one would still leave
  // another one reaching the hoisted code, making the rewrite invalid.
  if (Fallthrough.size() >= 2)
    return std::nullopt;

  // With a single fall-through branch, every other one is non-fallthrough, so
  // it is the only branch that can be hoisted.
  if (Fallthrough.size() == 1)
    return Fallthrough.front();

  // No branch falls through, so the whole operation is non-fallthrough and any
  // branch may be hoisted. The heuristic picks one.
  return selectByHeuristic(Regions);
}

// Moves the statements of the branch region \p R out of \p Op, into the nesting
// scope right after it. The emptied block is erased unless \p KeepBlock.
static void hoistBranchRegion(mlir::PatternRewriter &Rewriter,
                              mlir::Operation *Op,
                              mlir::Region &R,
                              bool KeepBlock) {
  mlir::Block *Block = getOnlyBlock(R);
  if (Block == nullptr)
    return;

  Rewriter.updateRootInPlace(Op, [&]() {
    inlineBlockBefore(Rewriter,
                      Block,
                      Op->getBlock(),
                      std::next(Op->getIterator()));
  });

  if (not KeepBlock)
    Rewriter.eraseBlock(Block);
}

// Selects the branch region of \p If to be hoisted, as an index into
// getBranchRegions(). If neither branch should be hoisted, the result is
// nullopt.
static std::optional<unsigned> selectIfHoistingTarget(IfOp If) {
  // Both empty region shapes are meaningful for an else, and are handled in
  // turn: a block-less region is an absent else clause, leaving nothing to
  // hoist, while an empty block is an else clause doing nothing, which is
  // dropped by hoisting it.
  if (If.getElse().empty())
    return std::nullopt;

  if (If.getElse().front().empty())
    return 1;

  // A then clause is instead always present, so its two empty region shapes
  // mean the same thing and are tested together. Inverting the condition drops
  // an empty then.
  if (isEmptyRegionOrBlock(If.getThen()))
    return 0;

  return selectHoistingTarget(If.getBranchRegions());
}

struct IfTerminalBranchComplementHoisting : mlir::OpRewritePattern<IfOp> {

  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(IfOp If, mlir::PatternRewriter &Rewriter) const override {
    std::optional<unsigned> Target = selectIfHoistingTarget(If);
    if (not Target)
      return mlir::failure();

    // Inverting an if hoisting its then branch keeps the readable
    // `if (!c) { ... }` form, and leaves the else as the only region to hoist.
    if (*Target == 0)
      invertIfStatement(Rewriter, If);

    hoistBranchRegion(Rewriter, If, If.getElse(), /*KeepBlock=*/false);
    return mlir::success();
  }
};

struct SwitchTerminalBranchComplementHoisting
  : mlir::OpRewritePattern<SwitchOp> {

  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SwitchOp Switch,
                  mlir::PatternRewriter &Rewriter) const override {
    llvm::MutableArrayRef<mlir::Region> Regions = Switch.getBranchRegions();

    std::optional<unsigned> Target = selectHoistingTarget(Regions);
    if (not Target)
      return mlir::failure();

    // An if drops an empty branch by hoisting it, but a switch has nothing to
    // gain from doing so.
    if (isEmptyRegionOrBlock(Regions[*Target]))
      return mlir::failure();

    // The branch regions of a switch are the default followed by the cases, and
    // only the emptied default is dropped. An emptied case keeps its now empty
    // body instead, so that its label still falls through to the hoisted code.
    //
    // Dropping a case is in fact never valid, which is subtle: a case is
    // hoisted only when the default does not fall through, either because the
    // case is the sole fall-through branch, or because the heuristic ran, which
    // requires every branch, the default included, to be non-fallthrough.
    // Removing the case would then route its value to that default rather than
    // to the hoisted code. A switch without a default hoists nothing anyway,
    // since its implicit fallthrough for unmatched values is itself a
    // fall-through branch, leaving no lone fall-through case to hoist.
    hoistBranchRegion(Rewriter,
                      Switch,
                      Regions[*Target],
                      /*KeepBlock=*/*Target != 0);
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
      Patterns.add<IfTerminalBranchComplementHoisting,
                   SwitchTerminalBranchComplementHoisting>(Context);

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
