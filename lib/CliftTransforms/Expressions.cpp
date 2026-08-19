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

struct CastCollapsingPattern
  : mlir::OpInterfaceRewritePattern<CastOpInterface> {

  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  struct CastRewriter {
    mlir::PatternRewriter &Rewriter;
    CastOpInterface Outer;
    CastOpInterface Inner;

    template<typename CastOpT>
    mlir::LogicalResult collapse() {
      mlir::Value Result = Outer.getResult();
      auto Op = Rewriter.create<CastOpT>(Outer->getLoc(),
                                         Result.getType(),
                                         Inner.getValue());

      Rewriter.replaceOp(Outer, { Op.getResult() });
      return mlir::success();
    }

    mlir::LogicalResult rewrite() {
      mlir::Type OuterT = Outer.getResult().getType();
      mlir::Type InnerT = Inner.getResult().getType();
      mlir::Type ValueT = Inner.getValue().getType();

      if (mlir::isa<BitCastOp>(Outer)) {
        if (mlir::isa<BitCastOp>(Inner))
          return collapse<BitCastOp>();

        if (not unwrapped_isa<IntegralType>(OuterT))
          return mlir::failure();

        if (mlir::isa<ExtendOp>(Inner))
          return collapse<ExtendOp>();

        if (mlir::isa<TruncateOp>(Inner))
          return collapse<TruncateOp>();

        return mlir::failure();
      }

      if (mlir::isa<ExtendOp>(Outer)) {
        if (isSigned(InnerT) != isSigned(ValueT))
          return mlir::failure();

        if (mlir::isa<BitCastOp>(Inner)) {
          if (not unwrapped_isa<IntegralType>(ValueT))
            return mlir::failure();

          return collapse<ExtendOp>();
        }

        if (mlir::isa<ExtendOp>(Inner))
          return collapse<ExtendOp>();

        return mlir::failure();
      }

      if (mlir::isa<TruncateOp>(Outer)) {
        if (mlir::isa<BitCastOp>(Inner)) {
          if (not unwrapped_isa<IntegralType>(ValueT))
            return mlir::failure();

          return collapse<TruncateOp>();
        }

        if (mlir::isa<ExtendOp>(Inner)) {
          auto SourceSize = getObjectSize(ValueT);
          auto TargetSize = getObjectSize(OuterT);

          if (TargetSize > SourceSize)
            return collapse<ExtendOp>();

          if (TargetSize < SourceSize)
            return collapse<TruncateOp>();

          return collapse<BitCastOp>();
        }

        if (mlir::isa<TruncateOp>(Inner))
          return collapse<TruncateOp>();

        return mlir::failure();
      }

      return mlir::failure();
    }
  };

  mlir::LogicalResult
  matchAndRewrite(CastOpInterface Outer,
                  mlir::PatternRewriter &Rewriter) const override {
    auto Inner = Outer.getValue().getDefiningOp<CastOpInterface>();
    if (not Inner)
      return mlir::failure();
    return CastRewriter(Rewriter, Outer, Inner).rewrite();
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
  Set.add<CastCollapsingPattern>(Set.getContext());
}

void clift::populateWithExpressionOptimizationPatterns(mlir::RewritePatternSet
                                                         &Set) {
  populateWithGenerated(Set);

  populateWithBooleanNegationPatterns(Set);
  populateWithCastCanonicalizations(Set);

  mlir::Dialect *Clift = Set.getContext()->getLoadedDialect<CliftDialect>();
  Clift->getCanonicalizationPatterns(Set);
}

PassPtr<FunctionOp> clift::createOptimizeExpressionsPass() {
  return std::make_unique<OptimizeExpressionsPass>();
}
