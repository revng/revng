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

static bool isSubjectToLvalueToRvalueConversion(mlir::OpOperand &Operand) {
  if (auto E = mlir::dyn_cast<ExpressionOpInterface>(Operand.getOwner()))
    return E.lvalueToRvalueConversion(Operand) == LvalueToRvalueConversion::Yes;

  revng_assert(mlir::isa<YieldOp>(Operand.getOwner()));

  // All yields are considered subject to l-value-to-r-value conversion.
  return true;
}

struct TypePunnedReadPattern : mlir::RewritePattern {
  TypePunnedReadPattern(mlir::MLIRContext *Context,
                        mlir::PatternBenefit Benefit = 1) :
    RewritePattern(MatchAnyOpTypeTag(), Benefit, Context) {}

  void initialize() { setDebugName("type-punned-read"); }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not mlir::isa<ExpressionOpInterface, YieldOp>(Op))
      return mlir::failure();

    mlir::LogicalResult Result = mlir::failure();
    for (mlir::OpOperand &Operand : Op->getOpOperands()) {
      if (not isSubjectToLvalueToRvalueConversion(Operand))
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

struct TypePunnedWritePattern : mlir::RewritePattern {
  TypePunnedWritePattern(mlir::MLIRContext *Context,
                         mlir::PatternBenefit Benefit = 1) :
    RewritePattern(MatchAnyOpTypeTag(), Benefit, Context) {}

  void initialize() { setDebugName("type-punned-write"); }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &Rewriter) const override {
    if (not mlir::isa<ExpressionOpInterface, YieldOp>(Op))
      return mlir::failure();

    mlir::LogicalResult Result = mlir::failure();
    for (mlir::OpOperand &Operand : Op->getOpOperands()) {
      bool IsDiscarded = isDiscardedOperand(Operand);
      if (not IsDiscarded and not isSubjectToLvalueToRvalueConversion(Operand))
        return mlir::failure();

      auto A = Operand.get().getDefiningOp<AssignOp>();
      if (not A)
        continue;

      auto I = A.getLhs().getDefiningOp<IndirectionOp>();
      if (not I)
        continue;

      auto C = I.getPointer().getDefiningOp<BitCastOp>();
      if (not C)
        continue;

      auto P1 = clift::unwrapped_dyn_cast<PointerType>(C.getValueType());
      if (not P1)
        continue;

      auto P2 = clift::unwrapped_cast<PointerType>(C.getType());

      auto T1 = P1.getPointeeType();
      auto T2 = P2.getPointeeType();

      if (getObjectSizeOrZero(T1) != getObjectSizeOrZero(T2))
        continue;

      // For now arrays are not value types and so cannot be bit-cast. In the
      // future, if array types are made regular, this rewrite can be applied.
      if (clift::unwrapped_isa<ArrayType>(T1)
          or clift::unwrapped_isa<ArrayType>(T2))
        continue;

      if (not isModifiableType(T1))
        continue;

      Rewriter.setInsertionPoint(A);

      mlir::Value LHS = C.getValue();
      LHS = Rewriter.create<IndirectionOp>(I->getLoc(), LHS);

      mlir::Value RHS = A.getRhs();
      RHS = Rewriter.create<BitCastOp>(A->getLoc(), T1, RHS);

      A->getOpOperand(0).set(LHS);
      A->getOpOperand(1).set(RHS);
      A.getResult().setType(T1);

      if (not IsDiscarded) {
        Rewriter.setInsertionPoint(Operand.getOwner());
        Operand.set(Rewriter.create<BitCastOp>(Operand.getOwner()->getLoc(),
                                               T2,
                                               A));
      }

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
  Set.add<TypePunnedWritePattern>(Set.getContext());

  mlir::Dialect *Clift = Set.getContext()->getLoadedDialect<CliftDialect>();
  Clift->getCanonicalizationPatterns(Set);
}

PassPtr<FunctionOp> clift::createOptimizeExpressionsPass() {
  return std::make_unique<OptimizeExpressionsPass>();
}
