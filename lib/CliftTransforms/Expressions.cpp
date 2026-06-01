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
namespace cast_canonicalization {

template<typename ExtendOpT>
static mlir::Value makeCastOpImpl(mlir::OpBuilder &Builder,
                                  mlir::Value ArgumentValue,
                                  mlir::Value ReplacedValue) {
  mlir::Type TargetType = ReplacedValue.getType();
  mlir::Location Loc = ReplacedValue.getDefiningOp()->getLoc();

  uint64_t SourceSize = getObjectSize(ArgumentValue.getType());
  uint64_t TargetSize = getObjectSize(TargetType);

  if (TargetSize > SourceSize)
    return Builder.create<ExtendOpT>(Loc, TargetType, ArgumentValue);

  if (TargetSize < SourceSize)
    return Builder.create<TruncateOp>(Loc, TargetType, ArgumentValue);

  return Builder.create<BitCastOp>(Loc, TargetType, ArgumentValue);
}

#include "revng/CliftTransforms/CastCanonicalization.h.inc"

} // namespace cast_canonicalization

namespace expression_optimization {

static bool areAllBitsSet(llvm::APInt Value, mlir::Type Type) {
  return Value.trunc(getUnderlyingIntegerType(Type).getSize() * 8).isAllOnes();
}

static bool assignTypePunnedConstraint(mlir::Value Ptr, mlir::Value Value) {
  auto PtrType = clift::unwrapped_dyn_cast<PointerType>(Ptr.getType());
  if (not PtrType)
    return false;

  mlir::Type SrcType = Value.getType();
  mlir::Type DstType = PtrType.getPointeeType();

  if (not clift::unwrapped_isa<ValueType>(DstType))
    return false;

  if (not isModifiableType(DstType))
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

} // namespace expression_optimization

struct TypePunnedReadPattern : mlir::RewritePattern {
  TypePunnedReadPattern(mlir::MLIRContext *Context,
                        mlir::PatternBenefit Benefit = 1) :
    RewritePattern(MatchAnyOpTypeTag(), Benefit, Context) {}

  static LvalueToRvalueConversion
  lvalueToRvalueConversion(mlir::OpOperand &Operand) {
    if (auto E = mlir::dyn_cast<ExpressionOpInterface>(Operand.getOwner()))
      return E.lvalueToRvalueConversion(Operand);

    revng_assert(mlir::isa<YieldOp>(Operand.getOwner()));

    // All yields are considered subject to l-value-to-r-value conversion.
    return LvalueToRvalueConversion::Yes;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not mlir::isa<ExpressionOpInterface, YieldOp>(Op))
      return mlir::failure();

    mlir::LogicalResult Result = mlir::failure();
    for (mlir::OpOperand &Operand : Op->getOpOperands()) {
      if (lvalueToRvalueConversion(Operand) != LvalueToRvalueConversion::Yes)
        continue;

      mlir::OpOperand *InnerOperand = &Operand;
      while (auto A = InnerOperand->get().getDefiningOp<AccessOp>()) {
        if (A.isIndirect())
          break;
        InnerOperand = &A->getOpOperand(0);
      }

      auto I = InnerOperand->get().getDefiningOp<IndirectionOp>();
      if (not I)
        continue;

      auto C = I.getPointer().getDefiningOp<BitCastOp>();
      if (not C)
        continue;

      auto P1 = clift::unwrapped_dyn_cast<PointerType>(C.getValueType());
      if (not P1)
        continue;

      auto P2 = clift::unwrapped_cast<PointerType>(I.getPointer().getType());

      auto T1 = P1.getPointeeType();
      auto T2 = P2.getPointeeType();

      if (getObjectSizeOrZero(T1) != getObjectSizeOrZero(T2))
        continue;

      // For now arrays are not value types and so cannot be bit-cast. In the
      // future, if array types are made regular, this rewrite can be applied.
      if (clift::unwrapped_isa<ArrayType>(T1)
          or clift::unwrapped_isa<ArrayType>(T2))
        continue;

      mlir::Operation *InnerOp = InnerOperand->getOwner();
      mlir::Value Value = C.getValue();

      Rewriter.setInsertionPoint(InnerOp);
      Value = Rewriter.create<IndirectionOp>(I.getLoc(), T1, Value);
      Value = Rewriter.create<BitCastOp>(InnerOp->getLoc(),
                                         removeConst(T2),
                                         Value);

      Rewriter.updateRootInPlace(InnerOp, [&]() { InnerOperand->set(Value); });

      Result = mlir::success();
    }
    return Result;
  }
};

struct OptimizeExpressionsPass
  : impl::CliftOptimizeExpressionsBase<OptimizeExpressionsPass> {

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);

    populateWithExpressionOptimizationPatterns(Set);
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

void clift::populateWithCastCanonicalizations(mlir::RewritePatternSet &Set) {
  cast_canonicalization::populateWithGenerated(Set);
}

void clift::populateWithExpressionOptimizationPatterns(mlir::RewritePatternSet
                                                         &Set) {
  expression_optimization::populateWithGenerated(Set);

  populateWithBooleanNegationPatterns(Set);
  populateWithCastCanonicalizations(Set);

  Set.add<TypePunnedReadPattern>(Set.getContext());

  mlir::Dialect *Clift = Set.getContext()->getLoadedDialect<CliftDialect>();
  Clift->getCanonicalizationPatterns(Set);
}

PassPtr<FunctionOp> clift::createOptimizeExpressionsPass() {
  return std::make_unique<OptimizeExpressionsPass>();
}
