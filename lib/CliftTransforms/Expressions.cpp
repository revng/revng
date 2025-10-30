//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/BooleanNegations.h"
#include "revng/CliftTransforms/ExpressionHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTOPTIMIZEEXPRESSIONS
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;
using namespace clift;

namespace {

static bool areAllBitsSet(llvm::APInt Value, mlir::Type Type) {
  return Value.trunc(getUnderlyingIntegerType(Type).getSize() * 8).isAllOnes();
}

static uint64_t truncateIntegerValue(mlir::IntegerAttr ValueAttr,
                                     mlir::Value IntegerOperand) {
  auto ValueType = mlir::cast<clift::ValueType>(IntegerOperand.getType());
  auto T = mlir::cast<PrimitiveType>(dealias(ValueType, true));

  uint64_t Value = ValueAttr.getValue().getZExtValue();
  return Value & (static_cast<uint64_t>(-1) >> (64 - 8 * T.getSize()));
}

static bool isCollapsibleCastKind(CastKind Kind) {
  return Kind != CastKind::Convert;
}

static bool assignTypePunnedConstraint(mlir::Value Ptr, mlir::Value Value) {
  auto PtrType = mlir::dyn_cast<PointerType>(Ptr.getType());
  if (not PtrType)
    return false;

  auto SrcType = mlir::cast<clift::ValueType>(Value.getType());
  auto DstType = PtrType.getPointeeType();

  return SrcType != DstType and SrcType.getByteSize() == DstType.getByteSize();
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

    auto NewPointerType = mlir::cast<PointerType>(PointerCast.getType());
    auto OldPointerType = PointerType::get(NewAssignment.getType(),
                                           NewPointerType.getPointerSize());

    Result = Rewriter.create<AddressofOp>(IndirectionLoc,
                                          OldPointerType,
                                          Result);

    Result = Rewriter.create<CastOp>(PointerCastLoc,
                                     NewPointerType,
                                     Result,
                                     CastKind::Bitcast);

    Result = Rewriter.create<IndirectionOp>(IndirectionLoc, Result);
  }
  return Result;
}

static bool hasEnumeratorValue(clift::ValueType Type, uint64_t Value) {
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
  auto PointerType = clift::getPointerType(PointerOperand.getType());
  revng_assert(PointerType);

  uint64_t Offset = OffsetAttr.getValue().getZExtValue();
  uint64_t Size = PointerType.getPointeeType().getByteSize();

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

struct OptimizeExpressionsPass
  : impl::CliftOptimizeExpressionsBase<OptimizeExpressionsPass> {

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);
    populateWithGenerated(Set);
    populateWithBooleanNegationPatterns(Set);

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
