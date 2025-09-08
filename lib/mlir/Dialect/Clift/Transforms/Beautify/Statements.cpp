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
#define GEN_PASS_DEF_CLIFTBEAUTIFYSTATEMENTS
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

using clift::BlockPosition;

static bool isConstantCondition(mlir::Region &R,
                                std::optional<bool> Value = std::nullopt) {
  if (auto Yield = clift::getYieldOp(R)) {
    if (auto Immediate = Yield.getValue().getDefiningOp<clift::ImmediateOp>())
      return not Value or static_cast<bool>(Immediate.getValue()) == *Value;
  }
  return false;
}

#if 0
class FallthroughRange {
  class sentinel {
  public:
    explicit sentinel() = default;
  };

  class iterator {
  public:
    explicit iterator(mlir::Block::iterator Pos) : Pos(Pos) {}

    mlir::Operation &operator*() const {
      revng_assert(Op != nullptr);
      return *Op;
    }

    iterator &operator++() & {
      revng_assert(Op != nullptr);

      auto Pos = BlockPosition::get(Op);
      auto &[B, I] = Pos;

      while (I == B->end()) {
        mlir::Operation *ParentOp = B->getParentOp();
        if (mlir::isa<clift::FunctionOp, clift::LoopOpInterface>(ParentOp))
          return setEnd();

        
      }

      Op = Pos.getOperation();
      return *this;
    }

    [[nodiscard]] iterator operator++(int) & {
      iterator It = *this;
      ++*this;
      return It;
    }

    friend bool operator==(iterator const& It, sentinel) {
      return Op == nullptr;
    }

  private:
    mlir::Operation *Op;

    iterator &setEnd() {
      Op = nullptr;
      return *this;
    }
  };

public:
  explicit FallthroughRange(mlir::Block *Block, mlir::Block::iterator Pos)
    : Block(Block), Pos(Pos) {}

  iterator begin() const {
    return iterator(Pos != Block->end() ? &*Pos : nullptr);
  }

  sentinel end() const { return sentinel(); }

private:
  mlir::Block *Block;
  mlir::Block::iterator Pos;
};

static FallthroughRange walkOperationsAfter(mlir::Operation *Op) {
  return FallthroughRange(std::next(Op->getIterator()));
}
#endif

[[maybe_unused]] // WIP
static void
removeBlock(mlir::Block *Block) {
  revng_assert(Block->getParent() != nullptr);
  Block->getParent()->getBlocks().remove(Block);
}

template<typename CallableT>
static void replaceExpression(mlir::PatternRewriter &Rewriter,
                              mlir::Region &Region,
                              CallableT &&Callable) {
  auto Yield = clift::getYieldOp(Region);
  revng_assert(Yield);

  mlir::OpBuilder::InsertionGuard Guard(Rewriter);
  Rewriter.setInsertionPoint(Yield.getOperation());

  mlir::Value Value = std::forward<CallableT>(Callable)(Yield.getValue());

  // WIP: Is this assign legal? Should we notify the rewriter somehow?
  Yield->getOpOperand(0).set(Value);
}

template<typename CallableT>
static void mergeExpressionInto(mlir::PatternRewriter &Rewriter,
                                mlir::Region &SourceRegion,
                                mlir::Region &TargetRegion,
                                CallableT &&Callable) {
  auto SourceYield = clift::getYieldOp(SourceRegion);
  revng_assert(SourceYield);

  auto TargetYield = clift::getYieldOp(TargetRegion);
  revng_assert(TargetYield);

  mlir::OpBuilder::InsertionGuard Guard(Rewriter);
  Rewriter.setInsertionPoint(TargetYield.getOperation());

  mlir::Value Value = std::forward<CallableT>(Callable)(SourceYield.getValue(),
                                                        TargetYield.getValue());

  // WIP: Is this assign legal? Should we notify the rewriter somehow?
  TargetYield->getOpOperand(0).set(Value);
}

static void moveBlocks(mlir::Region &Src, mlir::Region &Dst) {
  Dst.getBlocks().splice(Dst.end(), Src.getBlocks());
}

//===------------------ Future PatternRewriter functions ------------------===//

static void inlineBlockBefore(mlir::Block *Src,
                              mlir::Block *Dst,
                              mlir::Block::iterator Pos) {
  Dst->getOperations().splice(Pos, Src->getOperations());
}

#if 0
static void moveBlockBefore(mlir::PatternRewriter &Rewriter,
                            mlir::Block *Block,
                            mlir::Region *Region,
                            mlir::Region::iterator Pos) {
  revng_assert(Block->getParent() != Region);
  Rewriter.updateRootInPlace(If.getOperation(), [&]() {
    Region->getBlocks().splice(Pos,
                              Block->getParent()->getBlocks(),
                              Block->getIterator());
  });
}
#endif

static mlir::Type getBooleanType(mlir::MLIRContext *Context) {
  return clift::PrimitiveType::get(Context,
                                   clift::PrimitiveKind::SignedKind,
                                   1);
}

#if 0
[[maybe_unused]] // WIP
static bool
isJumpToStartOf(clift::GoToOp Goto, mlir::Region &Region) {
  if (Region.empty())
    return false;

  for (mlir::Operation &Op : Region.front()) {
    auto AssignLabel = mlir::dyn_cast<clift::AssignLabelOp>(&Op);

    if (not AssignLabel)
      break;

    if (Goto.getLabel() == AssignLabel.getLabel())
      return true;
  }

  return false;
}
#endif

#if 0
static BlockPosition findJumpTarget(mlir::Operation *Op) {
  if (mlir::isa<clift::ReturnOp>(Op)) {
    auto F = Op->getParentOfType<clift::FunctionOp>();
    return BlockPosition::getEnd(F.getBody());
  }

  if (mlir::isa<clift::LoopContinueOp>(Op)) {
    auto L = Op->getParentOfType<clift::LoopOpInterface>();
    return BlockPosition::getEnd(L.getLoopRegion());
  }

  if (mlir::isa<clift::SwitchBreakOp>(Op)) {
    auto S = Op->getParentOfType<clift::SwitchOp>();
    return getFallthroughTarget(BlockPosition::getNext(S.getOperation()));
  }

  if (auto G = mlir::dyn_cast<clift::GoToOp>(Op))
    return getGotoTarget(G);

  return {};
}
#endif

static BlockPosition skipLabels(BlockPosition Position) {
  if (Position) {
    auto &[B, I] = Position;
    while (I != B->end() and mlir::isa<clift::AssignLabelOp>(*I))
      ++I;
  }
  return Position;
}

static void invertBooleanExpression(mlir::PatternRewriter &Rewriter,
                                    mlir::Location Loc,
                                    mlir::Region &R) {
  replaceExpression(Rewriter, R, [&](mlir::Value Value) {
    auto BooleanType = getBooleanType(Rewriter.getContext());
    auto Op = Rewriter.create<clift::LogicalNotOp>(Loc, BooleanType, Value);
    return Op.getResult();
  });
}

static void invertIfStatement(mlir::PatternRewriter &Rewriter, clift::IfOp If) {
  mlir::Region *Then = &If.getThen();
  mlir::Region *Else = &If.getElse();
  revng_assert(not Else->empty());

  invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());

  Rewriter.updateRootInPlace(If.getOperation(), [&]() {
    mlir::Block *ThenBlock = Then->empty() ? nullptr : &Then->front();
    mlir::Block *ElseBlock = &Else->front();

    if (ThenBlock != nullptr)
      Then->getBlocks().remove(ThenBlock);

    Else->getBlocks().remove(ElseBlock);
    Then->getBlocks().push_back(ElseBlock);

    if (ThenBlock != nullptr)
      Else->getBlocks().push_back(ThenBlock);
  });
}

//===--------------------- Statement rewrite patterns ---------------------===//

#if 0
struct NestedIfCombiningPattern : mlir::RewritePattern {
  NestedIfCombiningPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.if", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    auto OuterIf = mlir::cast<clift::IfOp>(Root);
    auto InnerIf = getOnlyOperation<clift::IfOp>(OuterIf.getThen());
    if (not InnerIf) {
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if does not contain a nested clift.if.";
      });
    }

    mlir::Region &InnerElse = InnerIf.getElse();
    mlir::Region &OuterElse = OuterIf.getElse();

    if (not InnerElse.empty()) {
      if (OuterElse.empty()) {
        return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The inner clift.if has a non-empty else.";
        });
      }

      auto Goto = getOnlyOperation<clift::GoToOp>(InnerElse);
      if (not Goto) {
        return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The inner clift.if does not contain a nested clift.goto.";
        });
      }

      if (not isJumpToStartOf(Goto, OuterElse)) {
        return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
          Diag << "The clift.goto does not jump to the outer clift.if else.";
        });
      }
    } else if (not OuterElse.empty()) {
      return Rewriter.notifyMatchFailure(Root, [&](mlir::Diagnostic &Diag) {
        Diag << "The outer clift.if has a non-empty else.";
      });
    }

    mergeExpressionInto(Rewriter,
                        InnerIf.getCondition(),
                        OuterIf.getCondition(),
                        [&](mlir::Value InnerValue, mlir::Value OuterValue) {
      mlir::Location Loc = Rewriter.getFusedLoc(OuterIf.getLoc(),
                                                InnerIf.getLoc());

      auto BooleanType = getBooleanType(Rewriter.getContext());
      auto Op = Rewriter.create<clift::LogicalAndOp>(Loc,
                                                     BooleanType,
                                                     OuterValue,
                                                     InnerValue);

      return Op.getResult();
    });

    mlir::Region &InnerThen = InnerIf.getThen();
    mlir::Region &OuterThen = OuterIf.getThen();
    mlir::Block *InnerThenBlock = &InnerThen.front();

    // WIP: Should we notify the rewriter here?
    removeBlock(InnerThenBlock);
    Rewriter.eraseBlock(&OuterThen.front());
    OuterThen.push_back(InnerThenBlock);

    return mlir::success();
  }
};
#endif

struct LabelCombiningPattern : mlir::OpRewritePattern<clift::AssignLabelOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::AssignLabelOp AssignLabel,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Block::iterator Pos = std::next(AssignLabel->getIterator());

    if (Pos == AssignLabel->getBlock()->end())
      return mlir::failure();

    auto NextAssignLabel = mlir::dyn_cast<clift::AssignLabelOp>(&*Pos);
    if (not NextAssignLabel)
      return mlir::failure();

    Rewriter.replaceAllUsesWith(NextAssignLabel.getLabel(),
                                AssignLabel.getLabel());

    Rewriter.eraseOp(NextAssignLabel.getOperation());

    return mlir::success();
  }
};

struct BranchEqualizationPattern
  : mlir::OpInterfaceRewritePattern<clift::BranchOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::BranchOpInterface Branch,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Region *FallthroughRegion = nullptr;

    for (mlir::Region &R : Branch.getBranchRegions()) {
      // bool A = (bool)clift::getTrailingJumpOp(R);
      // bool B = not clift::hasIndirectFallthrough(R);
      // llvm::errs() << "A/B: " << A << "/" << B << "\n";

      if (not clift::hasIndirectFallthrough(R))
        continue;

      if (FallthroughRegion)
        return mlir::failure();

      FallthroughRegion = &R;
    }

    if (not FallthroughRegion)
      return mlir::failure();

    mlir::Block *Outer = Branch->getBlock();
    mlir::Block::iterator Beg = std::next(Branch->getIterator());
    mlir::Block::iterator End = Outer->end();

    // Skip backwards over any trailing labels:
    while (Beg != End and mlir::isa<clift::AssignLabelOp>(&*std::prev(End)))
      --End;

    if (Beg == End)
      return mlir::failure();

    if (mlir::cast<clift::StatementOpInterface>(std::prev(End))
          .hasIndirectFallthrough())
      return mlir::failure();

    if (FallthroughRegion->empty())
      FallthroughRegion->emplaceBlock();

    mlir::Block *Inner = &FallthroughRegion->front();

    Inner->getOperations().splice(Inner->end(),
                                  Outer->getOperations(),
                                  Beg,
                                  End);

    return mlir::success();
  }
};

struct EmptyIfInversionPattern : mlir::OpRewritePattern<clift::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::IfOp If,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not clift::isEmptyRegionOrBlock(If.getThen())) {
      return Rewriter.notifyMatchFailure(If, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has a non-empty then-region or block.";
      });
    }

    if (clift::isEmptyRegionOrBlock(If.getElse())) {
      return Rewriter.notifyMatchFailure(If, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has an empty else-region or block.";
      });
    }

    invertIfStatement(Rewriter, If);
    return mlir::success();
  }
};

struct EmptyElseEliminationPattern : mlir::OpRewritePattern<clift::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::IfOp If,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not clift::hasEmptyBlock(If.getElse())) {
      return Rewriter.notifyMatchFailure(If, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has a non-empty else-region.";
      });
    }

    Rewriter.eraseBlock(&If.getElse().front());
    return mlir::success();
  }
};

struct TerminalIfElseUnwrappingPattern : mlir::OpRewritePattern<clift::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::IfOp If,
                  mlir::PatternRewriter &Rewriter) const override {
    if (clift::isEmptyRegionOrBlock(If.getElse())) {
      return Rewriter.notifyMatchFailure(If, [&](mlir::Diagnostic &Diag) {
        Diag << "The clift.if has an empty else-region or block.";
      });
    }

    auto ThenFallthrough = clift::hasIndirectFallthrough(If.getThen());
    auto ElseFallthrough = clift::hasIndirectFallthrough(If.getElse());

    if (ThenFallthrough and ElseFallthrough) {
      return Rewriter.notifyMatchFailure(If, [&](mlir::Diagnostic &Diag) {
        Diag << "Both branches of the clift.if fall through.";
      });
    }

    if (not ThenFallthrough and not ElseFallthrough) {
      return Rewriter.notifyMatchFailure(If, [&](mlir::Diagnostic &Diag) {
        Diag << "Neither branch of the clift.if falls through.";
      });
    }

    if (ThenFallthrough)
      invertIfStatement(Rewriter, If);

    mlir::Block *ElseBlock = &If.getElse().front();
    Rewriter.updateRootInPlace(If.getOperation(), [&]() {
      inlineBlockBefore(ElseBlock,
                        If->getBlock(),
                        std::next(If->getIterator()));
    });

    Rewriter.eraseBlock(ElseBlock);
    return mlir::success();
  }
};

struct TrivialGotoEliminationPattern : mlir::OpRewritePattern<clift::GoToOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() { setDebugName("trivial-jump-elimination"); }

  static BlockPosition getFallthroughTarget(BlockPosition Position) {
    auto &[B, I] = Position;

    while (true) {
      Position = skipLabels(Position);

      if (I != B->end())
        break;

      mlir::Operation *ParentOp = B->getParentOp();
      if (not mlir::isa<clift::BranchOpInterface>(ParentOp))
        break;

      Position = BlockPosition::getNext(ParentOp);
    }

    return Position;
  }

  mlir::LogicalResult
  matchAndRewrite(clift::GoToOp Goto,
                  mlir::PatternRewriter &Rewriter) const override {
    auto GotoTarget = getFallthroughTarget(clift::getGotoTarget(Goto));
    auto FallTarget = getFallthroughTarget(BlockPosition::getNext(Goto));

    if (GotoTarget == FallTarget) {
      Rewriter.eraseOp(Goto);
      return mlir::success();
    }

#if 0
    if (auto FallGoto = FallTarget.getOperation<clift::GoToOp>()) {
      if (FallGoto.getLabel() == Goto.getLabel()) {
        Rewriter.eraseOp(Goto);
        return mlir::success();
      }
    }

    if (auto NextGoto = GotoTarget.getOperation<clift::GoToOp>()) {
      Rewriter.updateRootInPlace(Goto, [&]() {
        Goto.setOperand(NextGoto.getLabel());
      });

      return mlir::success();
    }
#endif

    return mlir::failure();
  }
};

struct WhileConditionHoistingPattern : mlir::OpRewritePattern<clift::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::WhileOp While,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not isConstantCondition(While.getCondition(), true))
      return mlir::failure();

    auto If = clift::getLeadingOp<clift::IfOp>(While.getBody());
    if (not If)
      return mlir::failure();

    auto ThenGoto = clift::getTrailingOp<clift::GoToOp>(If.getThen());
    auto ElseGoto = clift::getTrailingOp<clift::GoToOp>(If.getElse());

    if (not ThenGoto or not ElseGoto)
      return mlir::failure();

    auto BreakTarget = BlockPosition::getNext(While);
    bool IsThenBreak = getGotoTarget(ThenGoto) == BreakTarget;
    bool IsElseBreak = getGotoTarget(ElseGoto) == BreakTarget;

    if (IsThenBreak != IsElseBreak)
      return mlir::failure();

    if (IsThenBreak)
      invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());

    Rewriter.eraseOp(IsThenBreak ? ThenGoto : ElseGoto);
    mlir::Region &BreakRegion = IsThenBreak ? If.getThen() : If.getElse();
    mlir::Region &OtherRegion = IsThenBreak ? If.getElse() : If.getThen();

    While.getCondition().getBlocks().clear();
    moveBlocks(If.getCondition(), While.getCondition());

    auto &OuterOperations = While->getBlock()->getOperations();
    OuterOperations.splice(std::next(While->getIterator()),
                           BreakRegion.front().getOperations());

    auto &WhileOperations = While.getBody().front().getOperations();
    WhileOperations.splice(std::next(If->getIterator()),
                           OtherRegion.front().getOperations());

    Rewriter.eraseOp(If);

    return mlir::success();
  }
};

struct DoWhileConversionPattern : mlir::OpRewritePattern<clift::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::WhileOp While,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not isConstantCondition(While.getCondition(), true))
      return mlir::failure();

    auto If = clift::getTrailingOp<clift::IfOp>(While.getBody());
    if (not If or not If.getElse().empty())
      return mlir::failure();

    auto Last = clift::getTrailingOp<clift::StatementOpInterface>(If.getThen());
    if (Last.hasIndirectFallthrough())
      llvm::errs() << "fallthrough\n";
    if (Last.hasIndirectFallthrough())
      return mlir::failure();

    // With the break in the true branch, the condition must be inverted.
    invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());

    Rewriter.setInsertionPointAfter(While);
    auto DoWhile = Rewriter.create<clift::DoWhileOp>(While.getLoc());

    moveBlocks(If.getCondition(), DoWhile.getCondition());
    moveBlocks(While.getBody(), DoWhile.getBody());

    auto &OuterOperations = While->getBlock()->getOperations();
    OuterOperations.splice(std::next(DoWhile->getIterator()),
                           If.getThen().front().getOperations());

    Rewriter.eraseOp(While);
    Rewriter.eraseOp(If);

#if 0
    auto Goto = clift::getTrailingOp<clift::GoToOp>(If.getThen());
    if (not Goto or getGotoTarget(Goto) != BlockPosition::getNext(While))
      return mlir::failure();

    // With the break in the true branch, the condition must be inverted.
    invertBooleanExpression(Rewriter, If.getLoc(), If.getCondition());

    Rewriter.eraseOp(Goto);

    Rewriter.setInsertionPointAfter(While);
    auto DoWhile = Rewriter.create<clift::DoWhileOp>(While.getLoc());

    moveBlocks(If.getCondition(), DoWhile.getCondition());
    moveBlocks(While.getBody(), DoWhile.getBody());

    auto &OuterOperations = While->getBlock()->getOperations();
    OuterOperations.splice(std::next(DoWhile->getIterator()),
                           If.getThen().front().getOperations());

    Rewriter.eraseOp(While);
    Rewriter.eraseOp(If);

#endif
    return mlir::success();
  }
};

#if 0
struct OptimizedWhileConversionPattern : mlir::RewritePattern {
  OptimizedWhileConversionPattern(mlir::MLIRContext *Context) :
    // WIP: Think more about the benefit
    RewritePattern("clift.do_while", 3, Context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Root,
                  mlir::PatternRewriter &Rewriter) const override {
    
  }
};
#endif

struct BeautifyStatementsPass
  : clift::impl::CliftBeautifyStatementsBase<BeautifyStatementsPass> {
  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);

    Set.add(clift::MakeLabelOp::canonicalize);

    Set.add<LabelCombiningPattern>(Context);
    // Set.add<NestedIfCombiningPattern>(Context);
    Set.add<BranchEqualizationPattern>(Context);
    Set.add<EmptyIfInversionPattern>(Context);
    Set.add<EmptyElseEliminationPattern>(Context);
    Set.add<TerminalIfElseUnwrappingPattern>(Context);
    Set.add<TrivialGotoEliminationPattern>(Context);
    Set.add<DoWhileConversionPattern>(Context);

    Patterns = mlir::FrozenRewritePatternSet(std::move(Set),
                                             disabledPatterns,
                                             enabledPatterns);

    return mlir::success();
  }

  void runOnOperation() override {
    if (mlir::applyPatternsAndFoldGreedily(getOperation(), Patterns).failed())
      signalPassFailure();
  }

  mlir::FrozenRewritePatternSet Patterns;
};

} // namespace

clift::PassPtr<clift::FunctionOp> clift::createBeautifyStatementsPass() {
  return std::make_unique<BeautifyStatementsPass>();
}
