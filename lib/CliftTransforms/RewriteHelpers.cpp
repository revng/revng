//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/CliftTransforms/RewriteHelpers.h"

using namespace clift;

void clift::transformExpression(mlir::PatternRewriter &Rewriter,
                                mlir::Region &Region,
                                llvm::function_ref<mlir::Value(mlir::Value)>
                                  Transform) {
  auto Yield = clift::getYieldOp(Region);
  revng_assert(Yield);

  mlir::OpBuilder::InsertionGuard Guard(Rewriter);
  Rewriter.setInsertionPoint(Yield.getOperation());

  mlir::Value Value = Transform(Yield.getValue());

  Rewriter.updateRootInPlace(Region.getParentOp(),
                             [&]() { Yield->getOpOperand(0).set(Value); });
}

void clift::mergeExpressionInto(mlir::PatternRewriter &Rewriter,
                                mlir::Region &SourceRegion,
                                mlir::Region &TargetRegion,
                                llvm::function_ref<mlir::Value(mlir::Value,
                                                               mlir::Value)>
                                  Merge) {
  auto SourceYield = clift::getYieldOp(SourceRegion);
  revng_assert(SourceYield);

  auto TargetYield = clift::getYieldOp(TargetRegion);
  revng_assert(TargetYield);

  mlir::OpBuilder::InsertionGuard Guard(Rewriter);
  Rewriter.setInsertionPoint(TargetYield.getOperation());

  mlir::Value SourceValue = SourceYield.getValue();
  mlir::Value TargetValue = TargetYield.getValue();

  SourceYield->erase();
  inlineBlockBefore(Rewriter,
                    &SourceRegion.front(),
                    &TargetRegion.front(),
                    TargetYield->getIterator());

  mlir::Value Value = Merge(SourceValue, TargetValue);

  Rewriter.updateRootInPlace(TargetRegion.getParentOp(), [&]() {
    TargetYield->getOpOperand(0).set(Value);
  });
}

bool clift::isTriviallyTrue(mlir::Region &Condition) {
  if (auto Yield = clift::getYieldOp(Condition))
    return Yield.getValue().getDefiningOp<TrueOp>() != nullptr;

  return false;
}

void clift::invertBooleanExpression(mlir::PatternRewriter &Rewriter,
                                    mlir::Location Loc,
                                    mlir::Region &Region) {
  auto Transform = [&Rewriter, &Loc](mlir::Value Value) {
    return Rewriter.create<clift::LogicalNotOp>(Loc, Value);
  };
  transformExpression(Rewriter, Region, Transform);
}

void clift::invertIfStatement(mlir::PatternRewriter &Rewriter, IfOp If) {
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

static BlockPosition skipLabels(BlockPosition Position) {
  if (Position) {
    auto &[B, I] = Position;
    while (I != B->end() and mlir::isa<clift::AssignLabelOp>(*I))
      ++I;
  }
  return Position;
}

BlockPosition clift::getFallthroughTarget(BlockPosition Position) {
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
