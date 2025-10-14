#include <ranges>

#include "llvm/ADT/MapVector.h"

#include "mlir/Pass/Pass.h"

#include "revng/Support/Debug.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOpHelpers.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"
#include "revng/mlir/Dialect/Clift/Utils/LoopPromotion.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTLOOPPROMOTION
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

// Helper class to help in the `Loop` construction
class LoopConstructor {
private:
  clift::FunctionOp Func;
  mlir::OpBuilder Builder;

public:
  LoopConstructor(clift::FunctionOp F) : Func(F), Builder(F.getBody()) {}

  // Detects if a goto jumps backward to a label
  bool detectsBackwardJump(clift::GoToOp Jump, clift::AssignLabelOp Target) {
    mlir::Block *TargetBlock = Target->getBlock();
    mlir::Operation *Current = Jump.getOperation();

    // Traverse up through parent operations
    while (Current->getBlock() != TargetBlock) {
      mlir::Operation *Parent = Current->getParentOp();
      if (not mlir::isa<clift::BranchOpInterface>(Parent))
        return false;
      Current = Parent;
    }

    // Check if label appears before the jump in the same block
    auto JumpPos = Current->getIterator();
    auto LabelPos = Target->getIterator();
    auto BlockEnd = TargetBlock->end();

    for (auto Iter = LabelPos; Iter != JumpPos; ++Iter) {
      if (Iter == BlockEnd)
        return false;
    }

    return true;
  }

  // Extracts operations into a new block
  mlir::Block *splitBlockRange(mlir::Block *Source,
                               mlir::Block::iterator Start,
                               mlir::Block::iterator Finish) {
    auto *ExtractedBlock = new mlir::Block();
    ExtractedBlock->getOperations().splice(ExtractedBlock->begin(),
                                           Source->getOperations(),
                                           Start,
                                           Finish);
    return ExtractedBlock;
  }

  // Creates a new label operation
  mlir::Value generateLabel(mlir::Block *Target,
                            mlir::Block::iterator Position) {
    mlir::OpBuilder::InsertionGuard Guard(Builder);
    mlir::Location Loc = mlir::UnknownLoc::get(Func->getContext());

    Builder.setInsertionPointToStart(&Func.getBody().front());
    mlir::Value LabelValue = Builder.create<clift::MakeLabelOp>(Loc);

    Builder.setInsertionPoint(Target, Position);
    Builder.create<clift::AssignLabelOp>(Loc, LabelValue);

    return LabelValue;
  }

  // Determines the end position of the loop
  mlir::Block::iterator computeLoopBoundary(mlir::Operation *BoundOp,
                                            mlir::Block *EnclosingBlock) {
    while (BoundOp->getBlock() != EnclosingBlock)
      BoundOp = BoundOp->getParentOp();
    return std::next(BoundOp->getIterator());
  }

  // Builds the actual loop structure
  void buildLoopStructure(clift::AssignLabelOp Target,
                          llvm::ArrayRef<clift::GoToOp> BackwardJumps) {
    mlir::Location Loc = mlir::UnknownLoc::get(Func->getContext());
    mlir::Block *ContainingBlock = Target->getBlock();

    bool LastJumpInScope = (BackwardJumps.back()->getBlock()
                            == ContainingBlock);

    // Extract loop body
    mlir::Block::iterator BodyStart = std::next(Target->getIterator());
    mlir::Block::iterator BodyEnd = computeLoopBoundary(BackwardJumps.back(),
                                                        ContainingBlock);
    mlir::Block *LoopBody = splitBlockRange(ContainingBlock,
                                            BodyStart,
                                            BodyEnd);

    // Create `clift.while` loop operation
    Builder.setInsertionPoint(ContainingBlock,
                              std::next(Target->getIterator()));
    auto WhileLoop = Builder.create<clift::WhileOp>(Loc);

    // Setup loop condition (always `while(`)`), promotion will happen later
    Builder.setInsertionPointToStart(&WhileLoop.getCondition().emplaceBlock());
    auto
      IntegerType = clift::PrimitiveType::get(Builder.getContext(),
                                              clift::PrimitiveKind::SignedKind,
                                              4);
    auto TrueCondition = Builder.create<clift::ImmediateOp>(Loc,
                                                            IntegerType,
                                                            1);
    Builder.create<clift::YieldOp>(Loc, TrueCondition);

    // Push the newly constructed body into the `WhileOp`
    WhileLoop.getBody().push_back(LoopBody);

    // Handle termination, and insert `goto`s to mimic the `break`/`continue`
    // behavior
    llvm::ArrayRef<clift::GoToOp> RemainingJumps = BackwardJumps;

    if (LastJumpInScope) {
      revng_assert(BackwardJumps.back()->getBlock() == LoopBody);
      revng_assert(std::next(BackwardJumps.back()->getIterator())
                   == LoopBody->end());
      BackwardJumps.back()->erase();
      RemainingJumps = BackwardJumps.drop_back(1);
    } else {
      // Insert break for conditional gotos
      mlir::Value BreakTarget = generateLabel(ContainingBlock,
                                              std::next(WhileLoop
                                                          ->getIterator()));
      Builder.setInsertionPointToEnd(LoopBody);
      Builder.create<clift::GoToOp>(Loc, BreakTarget);
    }

    // Redirect remaining gotos to continue label
    if (not RemainingJumps.empty()) {
      mlir::Value ContinueTarget = generateLabel(LoopBody, LoopBody->end());
      for (clift::GoToOp Jump : RemainingJumps)
        Jump.setOperand(ContinueTarget);
    }
  }
};

// Core implementation which collects the backward `goto`s and promote them to
// `clift.loop`s
class LoopPromotionImpl {
private:
  using LabelGotoMap = llvm::MapVector<clift::AssignLabelOp,
                                       llvm::SmallVector<clift::GoToOp, 2>>;
  LabelGotoMap LabelJumpMapping;
  LoopConstructor Constructor;

public:
  LoopPromotionImpl(clift::FunctionOp F) : Constructor(F) {}

  // Perform a single walk on the `FunctionOp` and collect all the `Labels
  void collectLabelsAndGotos(clift::FunctionOp F) {
    F->walk([&](mlir::Operation *Op) {
      if (auto LabelAssign = mlir::dyn_cast<clift::AssignLabelOp>(Op)) {
        LabelJumpMapping.insert({ LabelAssign, {} });
      } else if (auto Jump = mlir::dyn_cast<clift::GoToOp>(Op)) {
        auto Entry = LabelJumpMapping.find(Jump.getAssignLabelOp());
        if (Entry != LabelJumpMapping.end()) {
          if (Constructor.detectsBackwardJump(Jump, Entry->first))
            Entry->second.push_back(Jump);
        }
      }
    });
  }

  void transformToLoops() {
    for (const auto &[Label, Jumps] : std::views::reverse(LabelJumpMapping)) {
      if (not Jumps.empty())
        Constructor.buildLoopStructure(Label, Jumps);
    }
  }
};

struct LoopPromotionPass
  : clift::impl::CliftLoopPromotionBase<LoopPromotionPass> {

  void runOnOperation() override {
    getOperation()->walk([](clift::FunctionOp Function) {
      inspectLoops(Function);
    });
  }
};

} // namespace

void mlir::clift::inspectLoops(clift::FunctionOp Function) {
  LoopPromotionImpl Impl(Function);
  Impl.collectLabelsAndGotos(Function);
  Impl.transformToLoops();
}

clift::PassPtr<clift::FunctionOp> clift::createLoopPromotionPass() {
  return std::make_unique<LoopPromotionPass>();
}
