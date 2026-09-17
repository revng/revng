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
#define GEN_PASS_DEF_CLIFTTERMINALBRANCHHOISTING
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

static bool hasInterveningSwitch(mlir::Operation *Op,
                                 mlir::Operation *Ancestor) {

  while (true) {
    mlir::Operation *ParentOp = Op->getParentOp();

    if (ParentOp == Ancestor)
      return false;

    if (mlir::isa<SwitchOp>(ParentOp))
      return true;

    Op = ParentOp;
  }
}

struct RegionWeight {
  unsigned LoopBreakCount;
  unsigned ApproximateLineCount;

  [[nodiscard]] friend auto operator<=>(const RegionWeight &LHS,
                                        const RegionWeight &RHS) = default;
};

class RegionWeightApproximator {
  mlir::Region &Region;

  unsigned LoopBreakCount = 0;
  unsigned ApproximateLineCount = 0;

public:
  [[nodiscard]] static RegionWeight approximate(mlir::Region &R) {
    return RegionWeightApproximator(R).approximate();
  }

private:
  explicit RegionWeightApproximator(mlir::Region &R) : Region(R) {}

  void approximateStatementWeight(mlir::Operation *Op) {
    // Statement scoped variables do not introduce an extra line:
    if (auto Local = mlir::dyn_cast<LocalVariableOp>(Op)) {
      if (isStatementScopedVariable(Local))
        return;
    }

    // Each statement spans at least one line:
    ++ApproximateLineCount;

    if (mlir::isa<LoopOpInterface>(Op)) {
      // Each loop statement is assumed to span one extra line due to the
      // closing brace:
      ++ApproximateLineCount;
    } else if (auto If = mlir::dyn_cast<IfOp>(Op)) {
      // Each if statement is assumed to span one extra line due to the closing
      // brace:
      ++ApproximateLineCount;

      // Each if statement with a non-empty else branch is assumed to span one
      // extra line due to the else:
      if (not If.getElse().empty())
        ++ApproximateLineCount;
    } else if (auto Switch = mlir::dyn_cast<SwitchOp>(Op)) {
      // Each switch statement is assumed to span one extra line due to the
      // closing brace:
      ++ApproximateLineCount;

      // Each case spans one line:
      ApproximateLineCount += Switch.getCaseCount();

      // Each case region is assumed to span one extra line due to the closing
      // brace:
      ApproximateLineCount += Switch.getCaseRegionCount();

      // The default case is assumed to span two lines due to the case label and
      // the closing brace:
      if (Switch.hasDefaultCase())
        ApproximateLineCount += 2;
    }
  }

  void detectLoopBreakInSwitch(mlir::Operation *Op) {
    auto Break = mlir::dyn_cast<BreakToOp>(Op);
    if (not Break)
      return;

    mlir::Operation *Branch = Region.getParentOp();
    if (not mlir::isa<SwitchOp>(Branch))
      return;

    mlir::Operation *Loop = Break.getLabelAssignmentOp();
    if (not Loop->isAncestor(Branch))
      return;

    if (hasInterveningSwitch(Break, Branch))
      return;

    if (hasInterveningSwitch(Branch, Loop))
      return;

    ++LoopBreakCount;
  }

  void visitStatement(StatementOpInterface Statement) {
    if (mlir::isa<MakeLabelOp, RequireOp>(Statement))
      return;

    detectLoopBreakInSwitch(Statement);
    approximateStatementWeight(Statement);
  }

  RegionWeight approximate() {
    // A single jump statement is treated as spanning zero lines, in order to
    // deprioritize its hoisting compared to any other sequence of statements.
    if (auto Jump = getOnlyOp<JumpStatementOpInterface>(Region)) {
      detectLoopBreakInSwitch(Jump);
    } else {
      Region.walk([this](StatementOpInterface Op) { visitStatement(Op); });
    }

    return { LoopBreakCount, ApproximateLineCount };
  }
};

struct TerminalBranchHoistingPattern
  : mlir::OpInterfaceRewritePattern<BranchOpInterface> {

  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(BranchOpInterface Branch,
                  mlir::PatternRewriter &Rewriter) const override {

    llvm::SmallVector<RegionWeight> Weights;
    for (mlir::Region &R : Branch.getBranchRegions()) {
      if (indirectlyFallsThrough(R))
        return mlir::failure();

      Weights.push_back(RegionWeightApproximator::approximate(R));
    }

    auto MaxIterator = std::ranges::max_element(Weights);
    auto MaxIndex = static_cast<unsigned>(MaxIterator - Weights.begin());

    hoistBranchRegion(Rewriter, Branch.getBranchRegions()[MaxIndex]);
    return mlir::success();
  }
};

struct TerminalBranchHoistingPass
  : clift::impl::CliftTerminalBranchHoistingBase<TerminalBranchHoistingPass> {

  mlir::FrozenRewritePatternSet Patterns;

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);
    Set.add<TerminalBranchHoistingPattern>(Context);
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

PassPtr<FunctionOp> clift::createTerminalBranchHoistingPass() {
  return std::make_unique<TerminalBranchHoistingPass>();
}
