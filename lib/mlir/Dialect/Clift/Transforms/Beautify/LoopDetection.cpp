//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"

#include "revng/mlir/Dialect/Clift/IR/CliftOpHelpers.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTLOOPDETECTION
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

static bool isBackwardGoto(clift::GoToOp Goto, clift::AssignLabelOp Label) {
  mlir::Block *LB = Label->getBlock();

  mlir::Operation *Op = Goto.getOperation();
  while (Op->getBlock() != LB) {
    mlir::Operation *ParentOp = Op->getParentOp();

    if (not mlir::isa<clift::BranchOpInterface>(ParentOp))
      return false;

    Op = ParentOp;
  }

  mlir::Block::iterator Pos = Op->getIterator();
  for (auto I = Label->getIterator(), E = LB->end(); I != Pos; ++I) {
    if (I == E)
      return false;
  }

  return true;
}

static size_t findBackwardGotos(clift::AssignLabelOp Label,
                                llvm::SmallVector<clift::GoToOp> &Gotos) {
  size_t TotalGotoCount = Gotos.size();

  for (auto UserOp : Label.getLabelOp()->getUsers()) {
    if (auto Goto = mlir::dyn_cast<clift::GoToOp>(UserOp)) {
      if (isBackwardGoto(Goto, Label))
        Gotos.push_back(Goto);
    }
  }

  return Gotos.size() - TotalGotoCount;
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

static void createLoop(clift::FunctionOp Function,
                       clift::AssignLabelOp LoopLabel,
                       llvm::ArrayRef<clift::GoToOp> Gotos) {
  mlir::Location LoopLoc = mlir::UnknownLoc::get(Function->getContext());

  mlir::Block *OuterBlock = LoopLabel->getBlock();

  // True if the last goto is in the scope of the label.
  bool GotoInLabelScope = Gotos.back()->getBlock() == OuterBlock;
  //llvm::errs() << "GotoInLabelScope: " << GotoInLabelScope << "\n";

  auto FindLoopEnd = [&](mlir::Operation *BoundingOp) -> mlir::Block::iterator {
    while (BoundingOp->getBlock() != OuterBlock)
      BoundingOp = BoundingOp->getParentOp();

    return std::next(BoundingOp->getIterator());
  };

  mlir::Block *InnerBlock = extractAsBlock(OuterBlock,
                                           std::next(LoopLabel->getIterator()),
                                           FindLoopEnd(Gotos.back()));

  mlir::OpBuilder Builder(Function.getBody());

  auto CreateLabel = [&](mlir::Block *Block, mlir::Block::iterator Pos) {
    mlir::OpBuilder::InsertionGuard Guard(Builder);

    Builder.setInsertionPointToStart(&Function.getBody().front());
    mlir::Value Label = Builder.create<clift::MakeLabelOp>(LoopLoc);

    Builder.setInsertionPoint(Block, Pos);
    Builder.create<clift::AssignLabelOp>(LoopLoc, Label);

    return Label;
  };

  Builder.setInsertionPoint(OuterBlock, std::next(LoopLabel->getIterator()));
  auto Loop = Builder.create<clift::WhileOp>(LoopLoc);

  Builder.setInsertionPointToStart(&Loop.getCondition().emplaceBlock());
  auto IntType = clift::PrimitiveType::get(Builder.getContext(),
                                           clift::PrimitiveKind::SignedKind,
                                           /*Size=*/1);
  Builder.create<clift::YieldOp>(LoopLoc,
                                 Builder.create<clift::ImmediateOp>(LoopLoc,
                                                                    IntType,
                                                                    1));

  Loop.getBody().push_back(InnerBlock);

  if (GotoInLabelScope) {
    // A loop-goto at the end of the loop body can be erased.
    revng_assert(Gotos.back()->getBlock() == InnerBlock);
    revng_assert(std::next(Gotos.back()->getIterator()) == InnerBlock->end());
    Gotos.back()->erase();

    // The last loop-goto has been erased and is no longer needed:
    Gotos = Gotos.drop_back(1);
  } else {
    // If the last loop-goto is conditional, a break must be inserted at the end
    // of the loop body.
    mlir::Value BreakLabel = CreateLabel(OuterBlock,
                                         std::next(Loop->getIterator()));

    Builder.setInsertionPointToEnd(InnerBlock);
    Builder.create<clift::GoToOp>(LoopLoc, BreakLabel);
  }

  // If there still exist any loop-gotos, a continue label is created at the end
  // of the loop body and the targets of all remaining loop-gotos are replaced.
  if (not Gotos.empty()) {
    mlir::Value ContinueLabel = CreateLabel(InnerBlock, InnerBlock->end());

    for (clift::GoToOp Goto : Gotos)
      Goto.setOperand(ContinueLabel);
  }

#if 0
  for (clift::GoToOp Goto : Gotos) {
    mlir::Operation *Op = Goto.getOperation();

    while (Op->getBlock() != InnerBlock) {
      mlir::Region *ParentRegion = Op->getParentRegion();
      mlir::Operation *ParentOp = ParentRegion->getParentOp();

      if (auto Branch = mlir::dyn_cast<clift::BranchOpInterface>(ParentOp)) {
        for (mlir::Region &R : Branch.getBranchRegions()) {
          if (&R == ParentRegion)
            continue;

          if (clift::getTrailingJumpOp(R))
            continue;

          if (R.empty())
            R.emplaceBlock();

          Builder.setInsertionPointToEnd(&R.front());
          Builder.create<clift::GoToOp>(LoopLoc, BreakLabel);
        }
      }

      Op = ParentOp;
    }
  }
#endif
}

static void createLoops(clift::FunctionOp Function) {
  llvm::SmallVector<std::pair<clift::AssignLabelOp, size_t>> LoopLabels;
  llvm::SmallVector<clift::GoToOp> LoopGotos;

  Function->walk([&](mlir::Operation *Op) {
    if (auto Label = mlir::dyn_cast<clift::AssignLabelOp>(Op)) {
      if (size_t const GotoCount = findBackwardGotos(Label, LoopGotos))
        LoopLabels.emplace_back(Label, GotoCount);
    }
  });

  for (size_t GotoIndex = 0; auto& [Label, GotoCount] : LoopLabels) {
    size_t I = std::exchange(GotoIndex, GotoIndex + GotoCount);
    createLoop(Function, Label, llvm::ArrayRef(LoopGotos).slice(I, GotoCount));
  }
}

struct LoopDetectionPass
  : clift::impl::CliftLoopDetectionBase<LoopDetectionPass> {

  void runOnOperation() override {
    getOperation()->walk([](clift::FunctionOp Function) {
      createLoops(Function);
    });
  }
};

} // namespace

clift::PassPtr<clift::FunctionOp> clift::createLoopDetectionPass() {
  return std::make_unique<LoopDetectionPass>();
}
