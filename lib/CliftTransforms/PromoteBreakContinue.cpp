//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"

#include "mlir/Pass/Pass.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTPROMOTEBREAKCONTINUE
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

/// A break_to can be promoted to a plain break when its target label is the
/// break label of the innermost enclosing loop and no switch statement is
/// nested between the break_to and that loop. Otherwise a plain break would
/// either target a different (inner) loop or be captured by an interposing
/// switch.
static bool canPromoteBreak(BreakToOp Break) {
  mlir::Value Label = Break.getLabel();
  if (not Label)
    return false;

  bool CrossedSwitch = false;
  LoopOpInterface Loop = getEnclosingLoop(Break.getOperation(), &CrossedSwitch);

  return Loop and not CrossedSwitch and Label == Loop.getBreakLabel();
}

/// A continue_to can be promoted to a plain continue when its target label is
/// the continue label of the innermost enclosing loop. Switch statements are
/// transparent to continue, so they need not be considered.
static bool canPromoteContinue(ContinueToOp Continue) {
  mlir::Value Label = Continue.getLabel();
  if (not Label)
    return false;

  LoopOpInterface Loop = getEnclosingLoop(Continue.getOperation());

  return Loop and Label == Loop.getContinueLabel();
}

static bool hasJumpUser(mlir::Value Label) {
  for (mlir::Operation *User : Label.getUsers()) {
    if (mlir::isa<JumpStatementOpInterface>(User))
      return true;
  }
  return false;
}

/// Once every break_to/continue_to referencing a loop label has been promoted,
/// the loop itself is the label's only remaining user and the label is dead:
/// the C backend would still emit it as an unreferenced label. Drop it from the
/// loop and erase the now-unused MakeLabelOp. This mirrors the cleanup
/// performed by MakeLabelOp::canonicalize.
static void removeDeadLoopLabel(LoopOpInterface Loop, bool IsBreak) {
  mlir::Value Label = IsBreak ? Loop.getBreakLabel() : Loop.getContinueLabel();
  if (not Label)
    return;

  auto Make = Label.getDefiningOp<MakeLabelOp>();
  if (not Make or hasJumpUser(Label))
    return;

  if (IsBreak)
    Loop.setBreakLabel(nullptr);
  else
    Loop.setContinueLabel(nullptr);

  Make->erase();
}

template<typename T>
using PassBase = clift::impl::CliftPromoteBreakContinueBase<T>;

struct PromoteBreakContinuePass : PassBase<PromoteBreakContinuePass> {

  void runOnOperation() override {
    // Promote eligible jumps by dropping their target label operand, and gather
    // the loops in the same walk. Clearing an operand does not change the IR
    // structure, so it is safe during the walk; dropping the now-dead labels
    // erases ops, so it is deferred until the walk has finished.
    llvm::SmallVector<LoopOpInterface> Loops;

    getOperation()->walk([&](mlir::Operation *Op) {
      if (auto Loop = mlir::dyn_cast<LoopOpInterface>(Op)) {
        Loops.push_back(Loop);
      } else if (auto Break = mlir::dyn_cast<BreakToOp>(Op)) {
        if (canPromoteBreak(Break))
          Break.getLabelMutable().clear();
      } else if (auto Continue = mlir::dyn_cast<ContinueToOp>(Op)) {
        if (canPromoteContinue(Continue))
          Continue.getLabelMutable().clear();
      }
    });

    for (LoopOpInterface Loop : Loops) {
      removeDeadLoopLabel(Loop, /*IsBreak=*/true);
      removeDeadLoopLabel(Loop, /*IsBreak=*/false);
    }
  }
};

} // namespace

PassPtr<FunctionOp> clift::createPromoteBreakContinuePass() {
  return std::make_unique<PromoteBreakContinuePass>();
}
