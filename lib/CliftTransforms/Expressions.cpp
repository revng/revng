//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/ExpressionHelpers.h"
#include "revng/CliftTransforms/Expressions.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTOPTIMIZEEXPRESSIONS
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

static bool areAllBitsSet(llvm::APInt Value, mlir::Type Type) {
  return Value.trunc(getUnderlyingIntegerType(Type).getSize() * 8).isAllOnes();
}

static uint64_t truncateIntegerValue(mlir::IntegerAttr ValueAttr,
                                     mlir::Value IntegerOperand) {
  uint64_t Width = getObjectSize(IntegerOperand.getType()) * 8;
  uint64_t Value = ValueAttr.getValue().getZExtValue();
  return Value & (static_cast<uint64_t>(-1) >> (64 - Width));
}

static bool assignTypePunnedConstraint(mlir::Value Ptr, mlir::Value Value) {
  auto PtrType = clift::unwrapped_dyn_cast<PointerType>(Ptr.getType());
  if (not PtrType)
    return false;

  mlir::Type SrcType = Value.getType();
  mlir::Type DstType = PtrType.getPointeeType();

  if (not clift::unwrapped_isa<ValueType>(DstType))
    return false;

  return SrcType != DstType
         and getObjectSize(SrcType) == getObjectSizeOrZero(DstType);
}

static mlir::Value assignTypePunnedResult(mlir::PatternRewriter &Rewriter,
                                          mlir::Value OldAssignment,
                                          mlir::Value NewAssignment,
                                          mlir::Value PointerCast,
                                          mlir::Value Indirection) {
  mlir::Value Result = NewAssignment;
  if (not isDiscarded(OldAssignment)) {
    mlir::Location PointerCastLoc = PointerCast.getDefiningOp()->getLoc();
    mlir::Location IndirectionLoc = Indirection.getDefiningOp()->getLoc();

    auto NewPtrType = mlir::cast<PointerType>(PointerCast.getType());
    auto OldPtrType = PointerType::get(NewAssignment.getType(),
                                       NewPtrType.getPointerSize());

    Result = Rewriter.create<AddressofOp>(IndirectionLoc, OldPtrType, Result);
    Result = Rewriter.create<BitCastOp>(PointerCastLoc, NewPtrType, Result);
    Result = Rewriter.create<IndirectionOp>(IndirectionLoc, Result);
  }
  return Result;
}

static bool hasEnumeratorValue(mlir::Type Type, uint64_t Value) {
  if (auto Enum = mlir::dyn_cast<EnumType>(Type)) {
    for (EnumFieldAttr Enumerator : Enum.getFields()) {
      if (Enumerator.getRawValue() == Value)
        return true;
    }
  }
  return false;
}

struct DivModPair {
  uint64_t Div;
  uint64_t Mod;
};

static DivModPair ptrOffsetDivMod(mlir::IntegerAttr OffsetAttr,
                                  mlir::Value PointerOperand) {
  auto PtrType = clift::unwrapped_cast<PointerType>(PointerOperand.getType());

  uint64_t Offset = OffsetAttr.getValue().getZExtValue();
  uint64_t Size = getObjectSizeOrZero(PtrType.getPointeeType());

  if (Size == 0) {
    return {
      .Div = 0,
      .Mod = static_cast<uint64_t>(-1),
    };
  }

  return {
    .Div = Offset / Size,
    .Mod = Offset % Size,
  };
}

#include "revng/CliftTransforms/Expressions.h.inc"

struct CastCollapsingPattern
  : mlir::OpInterfaceRewritePattern<CastOpInterface> {

  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(CastOpInterface Outer,
                  mlir::PatternRewriter &Rewriter) const override {
    auto Inner = Outer.getValue().getDefiningOp<CastOpInterface>();
    if (not Inner)
      return mlir::failure();

    if (Outer->getName() != Inner->getName())
      return mlir::failure();

    Rewriter.updateRootInPlace(Outer, [&]() {
      Outer->getOpOperand(0).set(Inner.getValue());
    });

    return mlir::failure();
  }
};

struct OptimizeExpressionsPass
  : impl::CliftOptimizeExpressionsBase<OptimizeExpressionsPass> {

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);
    populateWithGenerated(Set);
    populateWithBooleanNegationPatterns(Set);

    Set.add<CastCollapsingPattern>(Context);

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

    mlir::GreedyRewriteConfig Config;
    Config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
    if (mlir::applyPatternsAndFoldGreedily(Function, Patterns, Config).failed())
      signalPassFailure();
  }

  mlir::FrozenRewritePatternSet Patterns;
};

} // namespace

PassPtr<FunctionOp> clift::createOptimizeExpressionsPass() {
  return std::make_unique<OptimizeExpressionsPass>();
}
