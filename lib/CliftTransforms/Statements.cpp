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
#define GEN_PASS_DEF_CLIFTOPTIMIZESTATEMENTS
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

//===---------------------- Statement rewrite helpers ---------------------===//

/// Base class for patterns matching statement regions instead of specific
/// operations.
struct StatementRegionRewritePattern
  : mlir::OpInterfaceRewritePattern<StatementRegionOpInterface> {

  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  using OpInterfaceRewritePattern::matchAndRewrite;

  virtual mlir::LogicalResult
  matchAndRewrite(mlir::Region &Region,
                  mlir::PatternRewriter &Rewriter) const = 0;

  mlir::LogicalResult
  matchAndRewrite(StatementRegionOpInterface Op,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::LogicalResult Result = mlir::failure();
    for (mlir::Region &Region : Op.getStatementRegions()) {
      if (not isEmptyRegionOrBlock(Region)) {
        if (matchAndRewrite(Region, Rewriter).succeeded())
          Result = mlir::success();
      }
    }
    return Result;
  }
};

//===--------------------- Statement rewrite patterns ---------------------===//

/// Rewrites nested if-statements into conditions with conjunctions.
struct IfAndCombiningPattern : mlir::OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() { setDebugName("if-and-combining"); }

  mlir::LogicalResult
  matchAndRewrite(IfOp OuterIf,
                  mlir::PatternRewriter &Rewriter) const override {
    // This pattern attempts to match two nested if-statements. When both have
    // non-empty else branches, there are a total of four potentially matched
    // configurations. matchAndRewriteInner is thus invoked up to four times in
    // attempts to match each of the different configurations.

    if (matchAndRewriteOuter(OuterIf,
                             OuterIf.getThen(),
                             OuterIf.getElse(),
                             Rewriter)
          .succeeded())
      return mlir::success();

    if (matchAndRewriteOuter(OuterIf,
                             OuterIf.getElse(),
                             OuterIf.getThen(),
                             Rewriter)
          .succeeded())
      return mlir::success();

    return mlir::failure();
  }

  mlir::LogicalResult
  matchAndRewriteOuter(IfOp OuterIf,
                       mlir::Region &OuterBranch1,
                       mlir::Region &OuterBranch2,
                       mlir::PatternRewriter &Rewriter) const {
    if (auto InnerIf = getOnlyOp<IfOp>(OuterBranch1)) {
      if (matchAndRewriteInner(OuterIf,
                               OuterBranch1,
                               OuterBranch2,
                               InnerIf,
                               InnerIf.getThen(),
                               InnerIf.getElse(),
                               Rewriter)
            .succeeded())
        return mlir::success();

      if (matchAndRewriteInner(OuterIf,
                               OuterBranch1,
                               OuterBranch2,
                               InnerIf,
                               InnerIf.getElse(),
                               InnerIf.getThen(),
                               Rewriter)
            .succeeded())
        return mlir::success();
    }

    return mlir::failure();
  }

  /// \param OuterIf The outer if-statement.
  /// \param OuterBranch1 The region nesting \p InnerIf.
  /// \param OuterBranch2 The region opposite \p OuterBranch1.
  /// \param InnerIf The inner if-statement.
  /// \param InnerBranch1 The and-then region of the composed if-statement.
  /// \param InnerBranch2 The region opposite \p InnerBranch1.
  mlir::LogicalResult
  matchAndRewriteInner(IfOp OuterIf,
                       mlir::Region &OuterBranch1,
                       mlir::Region &OuterBranch2,
                       IfOp InnerIf,
                       mlir::Region &InnerBranch1,
                       mlir::Region &InnerBranch2,
                       mlir::PatternRewriter &Rewriter) const {
    if (OuterBranch2.empty() != InnerBranch2.empty())
      return mlir::failure();

    // If the outer branch has a non-empty opposing region, then the inner
    // branch must contain nothing other than a goto targeting that region.
    if (not OuterBranch2.empty()) {
      auto Goto = getOnlyOp<GotoOp>(InnerBranch2);
      if (not Goto)
        return mlir::failure();
      if (getJumpTarget(Goto) != BlockPosition::getBegin(OuterBranch2))
        return mlir::failure();
    }

    bool IsOuterElse = &OuterBranch1 == &OuterIf.getElse();
    bool IsInnerElse = &InnerBranch1 == &InnerIf.getElse();

    // If the and-then region is the inner else, and the nesting region of the
    // outer branch is not, or vice versa, the inner if condition is inverted.
    // This results in `C1 && !C2`, as opposed to `C1 && C2`.
    if (IsOuterElse ^ IsInnerElse) {
      invertBooleanExpression(Rewriter,
                              InnerIf.getLoc(),
                              InnerIf.getCondition());
    }

    auto Merge = [&](mlir::Value Inner, mlir::Value Outer) -> mlir::Value {
      auto BooleanType = getBooleanType(Rewriter.getContext());

      return Rewriter.create<LogicalAndOp>(InnerIf.getLoc(),
                                           BooleanType,
                                           Outer,
                                           Inner);
    };

    // Merge the two if conditions by joining the expressions together using a
    // logical and expression.
    mergeExpressionInto(Rewriter,
                        InnerIf.getCondition(),
                        OuterIf.getCondition(),
                        Merge);

    // Inline and-then region content into the nesting region of the outer
    // branch, and then finally remove the inner if statement entirely.
    if (not InnerBranch1.empty()) {
      inlineBlockBefore(Rewriter,
                        &InnerBranch1.front(),
                        &OuterBranch1.front(),
                        OuterBranch1.front().begin());
    }

    Rewriter.eraseOp(InnerIf);

    return mlir::success();
  }
};

/// Given a branch statement, out of whose branches only one falls through,
/// hoists code following the branch statement into its fallthrough region.
///
/// Example:
///   if (C) {
///     A;
///        <-------+
///   } else {     |
///     goto L;    | B; is moved into the first branch.
///   }            |
///   B; ----------+
///
/// While this rewrite may seem like it produces uglier code, it is important
/// for enabling other rewrites later. In any case, deciding on which branch to
/// hoist out requires heuristics and is done later in the pipeline.
struct BranchEqualizationPattern : StatementRegionRewritePattern {
  using StatementRegionRewritePattern::StatementRegionRewritePattern;

  void initialize() { setDebugName("branch-equalization"); }

  static mlir::Region *findUniqueFallthroughRegion(BranchOpInterface Branch) {
    mlir::Region *FallthroughRegion = nullptr;

    for (mlir::Region &R : Branch.getBranchRegions()) {
      if (clift::isIndirectlyNoFallthrough(R))
        continue;

      if (FallthroughRegion)
        return nullptr;

      FallthroughRegion = &R;
    }

    return FallthroughRegion;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Region &Region,
                  mlir::PatternRewriter &Rewriter) const override {
    revng_assert(Region.hasOneBlock());
    mlir::Block *Outer = &Region.front();

    auto Begin = Outer->begin();
    auto End = Outer->end();

    // Search backwards from the end of the region for a transformable branch
    // operation. Only one branch is transformed per each pattern application.
    while (Begin != End) {
      auto Branch = mlir::dyn_cast<BranchOpInterface>(&*--End);
      if (not Branch)
        continue;

      // Find the unique fallthrough region of the branch operation, if any.
      mlir::Region *FallthroughRegion = findUniqueFallthroughRegion(Branch);
      if (FallthroughRegion == nullptr)
        continue;

      mlir::Block::iterator B = std::next(End);
      mlir::Block::iterator E = Outer->end();

      // It is not desirable to hoist any trailing label assignments, as the
      // last valid location for a label assignment can be considered the most
      // canonical placement.
      //
      // Skip backwards over any trailing labels in the nesting region:
      while (B != E and mlir::isa<AssignLabelOp>(&*std::prev(E)))
        --E;

      // If sequence of operations to be hoisted is empty, or is composed of
      // only label assignments, there is no point in applying any change.
      if (B == E)
        continue;

      Rewriter.updateRootInPlace(Branch, [&]() {
        if (FallthroughRegion->empty())
          FallthroughRegion->emplaceBlock();

        // The selected statements are moved to the end of the fallthrough
        // region.
        mlir::Block *Inner = &FallthroughRegion->front();
        Inner->getOperations().splice(Inner->end(),
                                      Outer->getOperations(),
                                      B,
                                      E);
      });

      return mlir::success();
    }

    return mlir::failure();
  }
};

/// Inverts if-statements whose then-branches are empty.
struct EmptyIfInversionPattern : mlir::OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() { setDebugName("empty-if-inversion"); }

  mlir::LogicalResult
  matchAndRewrite(IfOp If, mlir::PatternRewriter &Rewriter) const override {
    if (not clift::isEmptyRegionOrBlock(If.getThen()))
      return mlir::failure();

    if (clift::isEmptyRegionOrBlock(If.getElse()))
      return mlir::failure();

    invertIfStatement(Rewriter, If);
    return mlir::success();
  }
};

/// Eliminates explicit jumps (goto, break_to, continue_to) where control would
/// fall through to the jump target anyway.
struct TrivialJumpEliminationPattern
  : mlir::OpInterfaceRewritePattern<StatementRegionOpInterface> {

  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  void initialize() { setDebugName("trivial-jump-elimination"); }

  using TrivialJumpVector = llvm::SmallVector<JumpStatementOpInterface>;

  /// Collects trivial jumps targeting a specific label. \p LabelPosition
  /// must point to the next non-label-assignment statement after the label
  /// described by \p Label.
  void findTrivialJumps(TrivialJumpVector &TrivialJumps,
                        mlir::Value Label,
                        BlockPosition LabelPosition) const {
    auto JumpTarget = getFallthroughTarget(LabelPosition);

    for (mlir::Operation *User : Label.getUsers()) {
      if (auto Jump = mlir::dyn_cast<JumpStatementOpInterface>(User)) {
        if (JumpTarget == getFallthroughTarget(BlockPosition::getNext(Jump)))
          TrivialJumps.push_back(Jump);
      }
    }
  }

  /// Collects trivial jumps targeting any labels whose target positions are
  /// nested directly within the specified region.
  void findTrivialJumps(TrivialJumpVector &TrivialJumps,
                        mlir::Region &Region) const {
    for (mlir::Operation &Op : Region.getOps()) {
      if (auto Assign = mlir::dyn_cast<AssignLabelOp>(&Op)) {
        findTrivialJumps(TrivialJumps,
                         Assign.getLabel(),
                         BlockPosition::getNext(Assign));
      } else if (auto Loop = mlir::dyn_cast<LoopOpInterface>(&Op)) {
        if (mlir::Value BreakLabel = Loop.getBreakLabel()) {
          findTrivialJumps(TrivialJumps,
                           BreakLabel,
                           BlockPosition::getNext(Loop));
        }
      }
    }
  }

  mlir::LogicalResult
  matchAndRewrite(StatementRegionOpInterface Op,
                  mlir::PatternRewriter &Rewriter) const override {
    // The matched operation of this pattern is any operation containing
    // statement regions. The set of jumps considered by the pattern is those
    // whose target is nested directly within a region of the target operation.
    //
    // All trivial jumps are collected before erasing any, in order to avoid
    // invalidating any iterators during the search.

    TrivialJumpVector TrivialJumps;

    // First, if the matched operation is a loop and it assigns a continue
    // label, then because that label logically targets a position directly
    // nested within the body region of the matched operation, trivial jumps
    // targeting the continue label are collected.
    if (auto Loop = mlir::dyn_cast<LoopOpInterface>(Op.getOperation())) {
      if (mlir::Value ContinueLabel = Loop.getContinueLabel()) {
        findTrivialJumps(TrivialJumps,
                         ContinueLabel,
                         BlockPosition::getEnd(Loop.getBody()));
      }
    }

    // Next, collect any trivial jumps targeting positions directly nested
    // within any statement region of the matched operation. This includes both
    // explicit label assignments as well as the break labels of any directly
    // nested loop operations, because break labels logically target positions
    // directly following their respective loop operations.
    for (mlir::Region &Region : Op.getStatementRegions())
      findTrivialJumps(TrivialJumps, Region);

    if (TrivialJumps.empty())
      return mlir::failure();

    // Finally, if any trivial jumps were found, they are erased.
    for (JumpStatementOpInterface Jump : TrivialJumps)
      Rewriter.eraseOp(Jump);

    return mlir::success();
  }
};

/// Converts while (1) statements with trailing conditional breaks into do-while
/// statements.
struct DoWhileConversionPattern : mlir::OpRewritePattern<WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() { setDebugName("do-while-conversion"); }

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

    // Check for an if-statement at the end of the loop body:
    auto If = clift::getLastOp<IfOp>(While.getBody());
    if (not If)
      return mlir::failure();

    auto IsBreak = [&While](mlir::Region &R) -> bool {
      auto Last = clift::getOnlyOp<BreakToOp>(R);
      return Last and Last.getLabelAssignmentOp() == While;
    };

    bool ThenBreak = IsBreak(If.getThen());
    bool ElseBreak = IsBreak(If.getElse());

    // If neither branch contains a break, this is not a do-while.
    if (not ThenBreak and not ElseBreak)
      return mlir::failure();

    // The region opposite the break-region must be empty.
    if (not isEmptyRegionOrBlock(ThenBreak ? If.getElse() : If.getThen()))
      return mlir::failure();

    if (ThenBreak) {
      // With the break in the true branch, the condition must be inverted.
      invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());
    }

    // The do-while loop is constructed after the while-loop, and its label
    // assignments are initialised by copying those of the while-loop.
    Rewriter.setInsertionPointAfter(While);
    auto DoWhile = Rewriter.create<DoWhileOp>(While.getLoc(), While);

    // The if-statement condition is inlined into the do-while condition.
    inlineRegionAtEnd(Rewriter, If.getCondition(), DoWhile.getCondition());

    // The while-loop body is inlined into do-while-loop body.
    inlineRegionAtEnd(Rewriter, While.getBody(), DoWhile.getBody());

    // Finally, the if-statement and while-loop - now empty - can be erased.
    Rewriter.eraseOp(While);
    Rewriter.eraseOp(If);

    return mlir::success();
  }
};

/// Insert a block statement directly nested in the region and move the existing
/// statements into the newly created block statement region.
static void wrapInBlockStatement(mlir::Region &R) {
  mlir::Block *OldBlock = &R.front();
  R.getBlocks().remove(OldBlock);

  mlir::Block *NewBlock = &R.emplaceBlock();
  mlir::OpBuilder Builder(NewBlock, NewBlock->end());

  auto S = Builder.create<BlockStatementOp>(R.getParentOp()->getLoc());
  S.getBlock().push_back(OldBlock);
}

/// Given a statement region containing only a single block statement, removes
/// the block statement and moves any statements nested within into the region
/// previously containing the block statement.
static void unwrapBlockStatement(mlir::Region &R) {
  auto S = clift::getOnlyOp<BlockStatementOp>(R);
  revng_assert(S);

  mlir::Block *Block = clift::extractOnlyBlock(S.getBlock());
  S->erase();

  clift::setOnlyBlock(R, Block);
}

struct OptimizeStatementsPass
  : clift::impl::CliftOptimizeStatementsBase<OptimizeStatementsPass> {

  mlir::FrozenRewritePatternSet Patterns;

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);

    Set.add(MakeLabelOp::canonicalize);

    Set.add<IfAndCombiningPattern>(Context);
    Set.add<BranchEqualizationPattern>(Context);
    Set.add<EmptyIfInversionPattern>(Context);
    Set.add<TrivialJumpEliminationPattern>(Context);
    Set.add<DoWhileConversionPattern>(Context);

    Patterns = mlir::FrozenRewritePatternSet(std::move(Set),
                                             disabledPatterns,
                                             enabledPatterns);

    return mlir::success();
  }

  void runOnOperation() override {
    FunctionOp Function = getOperation();
    mlir::Region &Body = Function.getBody();

    if (Body.empty())
      return;

    // Before applying rewrites, the function content is wrapped in a block
    // statement. This enables the logical root operation (previously function,
    // now the block statement) to be considered for rewriting. This is needed
    // in particular for region patterns (StatementRegionRewritePattern).
    wrapInBlockStatement(Body);

    mlir::GreedyRewriteConfig Config;
    Config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
    if (mlir::applyPatternsAndFoldGreedily(Function, Patterns, Config).failed())
      signalPassFailure();

    unwrapBlockStatement(Body);
  }
};

} // namespace

PassPtr<FunctionOp> clift::createOptimizeStatementsPass() {
  return std::make_unique<OptimizeStatementsPass>();
}
