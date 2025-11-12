//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <ranges>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"

#include "mlir/Pass/Pass.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTLOOPDETECTION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;
using namespace clift;

namespace {

// Needed for applying patterns locally to a single operation.
struct TrivialPatternRewriter : public mlir::PatternRewriter {
  explicit TrivialPatternRewriter(mlir::MLIRContext *Context) :
    PatternRewriter(Context) {}
};

template<typename IteratorT, typename SentinelT>
static bool isOrderedBefore(IteratorT First, IteratorT Second, SentinelT End) {
  for (; First != End; ++First) {
    if (First == Second)
      return true;
  }
  return false;
}

static bool isBackwardGoto(GotoOp Goto, AssignLabelOp Label) {
  mlir::Block *LabelBlock = Label->getBlock();
  mlir::Operation *CurrentOp = Goto.getOperation();

  // Traverse up through nesting branch operations until the current operation
  // and the target label operation are directly nested within the same block.
  while (CurrentOp->getBlock() != LabelBlock) {
    mlir::Operation *ParentOp = CurrentOp->getParentOp();

    // Any goto nested within a non-branch operation is conservatively
    // considered non-looping. Certainly encountering an existing loop or the
    // nesting function operation prevents the creation of a loop.
    if (not mlir::isa<BranchOpInterface>(ParentOp))
      return false;

    CurrentOp = ParentOp;
  }

  // It is known from the order of label and goto processing that the label must
  // precede the goto.
  revng_assert(isOrderedBefore(Label->getIterator(),
                               CurrentOp->getIterator(),
                               LabelBlock->end()));

  return true;
}

static mlir::Block *extractAsBlock(mlir::Block *Block,
                                   mlir::Block::iterator Begin,
                                   mlir::Block::iterator End) {
  mlir::Block *NewBlock = new mlir::Block();

  NewBlock->getOperations().splice(NewBlock->begin(),
                                   Block->getOperations(),
                                   Begin,
                                   End);

  return NewBlock;
}

static mlir::Block::iterator findLoopBoundary(mlir::Block *NestingBlock,
                                              mlir::Operation *BoundingOp) {
  while (BoundingOp->getBlock() != NestingBlock)
    BoundingOp = BoundingOp->getParentOp();

  return std::next(BoundingOp->getIterator());
}

static mlir::Value createLabel(mlir::OpBuilder &Builder,
                               FunctionOp Function,
                               mlir::Location Location) {
  mlir::OpBuilder::InsertionGuard Guard(Builder);
  Builder.setInsertionPointToStart(&Function.getBody().front());
  return Builder.create<MakeLabelOp>(Location);
}

/// Creates a loop starting after \p LoopLabel and bounded by the last goto in
/// \p Gotos (inclusive) if it is nested directly in the scope containing
/// \p LoopLabel, or by the operation nesting it in that scope.
///
/// The last goto is removed if possible. In any case, any remaining gotos
/// targeting \p LoopLabel are converted into continue-to operations.
///
/// If a label exists immediately after the newly created loop, any gotos within
/// the loop body targeting that label are converted into break-to operations.
///
/// The set of goto operations specified by \p Gotos are called looping-gotos.
static void createLoop(FunctionOp Function,
                       AssignLabelOp LoopLabel,
                       llvm::ArrayRef<GotoOp> Gotos,
                       llvm::DenseSet<MakeLabelOp> &AffectedLabels) {
  mlir::Location LoopLoc = Gotos.back()->getLoc();

  mlir::Block *LabelBlock = LoopLabel->getBlock();
  bool BoundingGotoIsInLabelScope = Gotos.back()->getBlock() == LabelBlock;

  auto BodyBegin = std::next(LoopLabel->getIterator());
  auto BodyEnd = findLoopBoundary(LabelBlock, Gotos.back());
  mlir::Block *BodyBlock = extractAsBlock(LabelBlock, BodyBegin, BodyEnd);

  mlir::OpBuilder Builder(Function.getBody());

  Builder.setInsertionPoint(LabelBlock, std::next(LoopLabel->getIterator()));
  auto Loop = Builder.create<WhileOp>(LoopLoc);

  Builder.setInsertionPointToStart(&Loop.getCondition().emplaceBlock());

  auto Bool = PrimitiveType::get(Builder.getContext(),
                                 PrimitiveKind::SignedKind,
                                 /*Size=*/1);
  auto True = Builder.create<ImmediateOp>(LoopLoc, Bool, 1);

  Builder.create<YieldOp>(LoopLoc, True);

  Loop.getBody().push_back(BodyBlock);

  if (BoundingGotoIsInLabelScope) {
    // A looping-goto at the end of the loop body can be erased. Leaving it
    // would not be particularly harmful, as the targeted continue-label would
    // immediately follow the jump, and it could be removed in a later pass, but
    // it might as well be removed eagerly here to make the loop detection
    // output slightly easier to read and to test.
    revng_assert(Gotos.back()->getBlock() == BodyBlock);
    revng_assert(std::next(Gotos.back()->getIterator()) == BodyBlock->end());
    Gotos.back()->erase();

    // The last looping-goto has been erased and is no longer needed:
    Gotos = Gotos.drop_back(1);
  } else {
    // If the last looping-goto is conditional, a break must be inserted at the
    // end of the loop body.

    mlir::Value BreakLabel = createLabel(Builder, Function, LoopLoc);
    Loop.setBreakLabel(BreakLabel);

    Builder.setInsertionPointToEnd(BodyBlock);
    Builder.create<BreakToOp>(LoopLoc, BreakLabel);
  }

  // If any looping-gotos still exist, they are replaced by continue-to
  // operations targeting the loop's continue-label.
  if (not Gotos.empty()) {
    mlir::Value ContinueLabel = createLabel(Builder, Function, LoopLoc);
    Loop.setContinueLabel(ContinueLabel);

    for (GotoOp Goto : Gotos) {
      Builder.setInsertionPoint(Goto);
      Builder.create<ContinueToOp>(Goto.getLoc(), ContinueLabel);
      Goto->erase();
    }
  }

  AffectedLabels.insert(LoopLabel.getLabelOp());

  // If there exists a label assignment immediately following the newly created
  // loop, any gotos targeting that label from within the loop are converted
  // into break-to operations targeting the loop's break-label.
  if (auto BreakAssignment = getNextOp<AssignLabelOp>(Loop)) {
    for (mlir::Operation *User : BreakAssignment.getLabel().getUsers()) {
      auto Goto = mlir::dyn_cast<GotoOp>(User);
      if (not Goto)
        continue;
      if (not Loop->isAncestor(Goto))
        continue;
      if (not Loop.getBreakLabel())
        Loop.setBreakLabel(createLabel(Builder, Function, LoopLoc));

      Builder.setInsertionPoint(Goto);
      Builder.create<BreakToOp>(Goto.getLoc(), Loop.getBreakLabel());
      Goto->erase();
    }

    if (not BreakAssignment.getLabel().use_empty())
      AffectedLabels.insert(BreakAssignment.getLabelOp());
  }

  // If there exists a label assignment at the end of the the newly created
  // loop, any gotos targeting that label from within the loop are converted
  // into continue-to operations targeting the loop's continue-label.
  //
  // Note that while any looping-gotos were already converted into continue-to
  // operations, it is important to check for this label as well. Failure to
  // convert gotos targeting this label into continue-to operations here would
  // result in verification failures later after label merging.
  if (auto ContinueAssignment = getLastOp<AssignLabelOp>(Loop.getBody())) {
    for (mlir::Operation *User : ContinueAssignment.getLabel().getUsers()) {
      auto Goto = mlir::dyn_cast<GotoOp>(User);
      if (not Goto)
        continue;
      if (not Loop->isAncestor(Goto))
        continue;
      if (not Loop.getContinueLabel())
        Loop.setContinueLabel(createLabel(Builder, Function, LoopLoc));

      Builder.setInsertionPoint(Goto);
      Builder.create<ContinueToOp>(Goto.getLoc(), Loop.getContinueLabel());
      Goto->erase();
    }

    if (not ContinueAssignment.getLabel().use_empty())
      AffectedLabels.insert(ContinueAssignment.getLabelOp());
  }
}

static void createLoops(FunctionOp Function) {
  // Maps each label assignment to the backwards gotos targeting it.
  llvm::MapVector<AssignLabelOp, llvm::SmallVector<GotoOp, 2>> Labels;

  // Walk the function in syntactic order (whether it is pre-order, in-order, or
  // post-order doesn't actually matter), searching for label assignment and
  // goto operations. Gotos whose targeted label assignment has not yet been
  // visited could not possibly represent a loop, and are thus ignored.
  //
  // All visited labels are recorded, along with a associated (possibly empty)
  // set of goto operations considered backwards (valid for loop creation).
  Function->walk([&](mlir::Operation *Op) {
    if (auto Label = mlir::dyn_cast<AssignLabelOp>(Op)) {
      Labels.insert({ Label, {} });
    } else if (auto Goto = mlir::dyn_cast<GotoOp>(Op)) {
      mlir::Operation *Assignment = Goto.getLabelAssignmentOp();
      if (auto A = mlir::dyn_cast<AssignLabelOp>(Assignment)) {
        if (auto Iterator = Labels.find(A); Iterator != Labels.end()) {
          if (isBackwardGoto(Goto, Iterator->first))
            Iterator->second.push_back(Goto);
        }
      }
    }
  });

  llvm::DenseSet<MakeLabelOp> AffectedLabels;

  // In order to process potentially nested loops before their nesting loops, it
  // is particularly important to iterate over the set of labels in reverse.
  for (const auto &[Label, Gotos] : std::views::reverse(Labels)) {
    if (not Gotos.empty())
      createLoop(Function, Label, Gotos, AffectedLabels);
  }

  // Canonicalize affected labels. This causes the label and any remaining
  // assignments to be erased if no jumps targeting the label exist anymore.
  if (not AffectedLabels.empty()) {
    TrivialPatternRewriter Rewriter(Function.getContext());
    for (MakeLabelOp AffectedLabel : AffectedLabels)
      (void) MakeLabelOp::canonicalize(AffectedLabel, Rewriter);
  }
}

struct LoopDetectionPass : impl::CliftLoopDetectionBase<LoopDetectionPass> {
  void runOnOperation() override { createLoops(getOperation()); }
};

} // namespace

PassPtr<FunctionOp> clift::createLoopDetectionPass() {
  return std::make_unique<LoopDetectionPass>();
}
