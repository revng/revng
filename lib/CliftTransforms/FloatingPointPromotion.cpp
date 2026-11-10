//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string_view>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTFLOATINGPOINTPROMOTION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

enum class ExternTypeKind : uint8_t {
  None,
  Bool,
  Float,
  Signed,
  Unsigned,
};

struct ExternType {
  ExternTypeKind Kind;
  uint8_t Size;

  [[nodiscard]] friend bool operator==(const ExternType &,
                                       const ExternType &) = default;
};

struct FloatingPointOperation {
  std::string_view Symbol;

  ExternType ReturnType;
  ExternType ParameterTypes[2];

  llvm::StringRef Operation;
};

static constexpr ExternType Bool = { ExternTypeKind::Bool, 0 };

static constexpr ExternType F32 = { ExternTypeKind::Float, 4 };
static constexpr ExternType F64 = { ExternTypeKind::Float, 8 };

static constexpr ExternType I16 = { ExternTypeKind::Signed, 2 };
static constexpr ExternType I32 = { ExternTypeKind::Signed, 4 };
static constexpr ExternType I64 = { ExternTypeKind::Signed, 8 };

static constexpr ExternType U16 = { ExternTypeKind::Unsigned, 2 };
static constexpr ExternType U32 = { ExternTypeKind::Unsigned, 4 };
static constexpr ExternType U64 = { ExternTypeKind::Unsigned, 8 };

#define N(Operation) (Operation::getOperationName())
static constexpr FloatingPointOperation FloatingPointOperations[] = {
  { "float32_add", F32, { F32, F32 }, N(FAddOp) },
  { "float32_chs", F32, { F32 }, N(FNegOp) },
  { "float32_div", F32, { F32, F32 }, N(FDivOp) },
  { "float32_eq_quiet", Bool, { F32, F32 }, N(FCmpEqOp) },
  { "float32_le", Bool, { F32, F32 }, N(FCmpLeOp) },
  { "float32_lt", Bool, { F32, F32 }, N(FCmpLtOp) },
  { "float32_mul", F32, { F32, F32 }, N(FMulOp) },
  { "float32_sub", F32, { F32, F32 }, N(FSubOp) },
  { "float32_to_float64", F64, { F32 }, N(FloatToFloatOp) },
  { "float32_to_int16_round_to_zero", I16, { F32 }, N(FloatToSignedOp) },
  { "float32_to_int32_round_to_zero", I32, { F32 }, N(FloatToSignedOp) },
  { "float32_to_int64_round_to_zero", I64, { F32 }, N(FloatToSignedOp) },
  { "float32_to_uint16_round_to_zero", U16, { F32 }, N(FloatToUnsignedOp) },
  { "float32_to_uint32_round_to_zero", U32, { F32 }, N(FloatToUnsignedOp) },
  { "float32_to_uint64_round_to_zero", U64, { F32 }, N(FloatToUnsignedOp) },
  { "float64_add", F64, { F64, F64 }, N(FAddOp) },
  { "float64_chs", F64, { F64 }, N(FNegOp) },
  { "float64_div", F64, { F64, F64 }, N(FDivOp) },
  { "float64_eq_quiet", Bool, { F32, F32 }, N(FCmpEqOp) },
  { "float64_le", Bool, { F32, F32 }, N(FCmpLeOp) },
  { "float64_lt", Bool, { F32, F32 }, N(FCmpLtOp) },
  { "float64_mul", F64, { F64, F64 }, N(FMulOp) },
  { "float64_sub", F64, { F64, F64 }, N(FSubOp) },
  { "float64_to_float32", F32, { F64 }, N(FloatToFloatOp) },
  { "float64_to_int16_round_to_zero", I16, { F64 }, N(FloatToSignedOp) },
  { "float64_to_int32_round_to_zero", I32, { F64 }, N(FloatToSignedOp) },
  { "float64_to_int64_round_to_zero", I64, { F64 }, N(FloatToSignedOp) },
  { "float64_to_uint16_round_to_zero", U16, { F64 }, N(FloatToUnsignedOp) },
  { "float64_to_uint32_round_to_zero", U32, { F64 }, N(FloatToUnsignedOp) },
  { "float64_to_uint64_round_to_zero", U64, { F64 }, N(FloatToUnsignedOp) },
  { "int16_to_float32", F32, { I16 }, N(SignedToFloatOp) },
  { "int16_to_float64", F64, { I32 }, N(SignedToFloatOp) },
  { "int32_to_float32", F32, { I64 }, N(SignedToFloatOp) },
  { "int32_to_float64", F64, { I16 }, N(SignedToFloatOp) },
  { "int64_to_float32", F32, { I32 }, N(SignedToFloatOp) },
  { "int64_to_float64", F64, { I64 }, N(SignedToFloatOp) },
  { "uint16_to_float32", F32, { U16 }, N(UnsignedToFloatOp) },
  { "uint16_to_float64", F64, { U32 }, N(UnsignedToFloatOp) },
  { "uint32_to_float32", F32, { U64 }, N(UnsignedToFloatOp) },
  { "uint32_to_float64", F64, { U16 }, N(UnsignedToFloatOp) },
  { "uint64_to_float32", F32, { U32 }, N(UnsignedToFloatOp) },
  { "uint64_to_float64", F64, { U64 }, N(UnsignedToFloatOp) },
};
#undef N

static_assert(std::ranges::is_sorted(FloatingPointOperations,
                                     std::less(),
                                     &FloatingPointOperation::Symbol));

static ExternType getExternType(mlir::Type Type) {
  auto TruncateSize = [](uint64_t Size) -> uint8_t {
    revng_assert(Size <= std::numeric_limits<uint8_t>::max());
    return static_cast<uint8_t>(Size);
  };

  if (auto T = mlir::dyn_cast<FloatType>(Type))
    return { ExternTypeKind::Float, TruncateSize(T.getSize()) };

  if (auto T = mlir::dyn_cast<IntegerType>(Type)) {
    if (T.getKind() == IntegerKind::Signed)
      return { ExternTypeKind::Signed, TruncateSize(T.getSize()) };

    if (T.getKind() == IntegerKind::Unsigned)
      return { ExternTypeKind::Signed, TruncateSize(T.getSize()) };
  }

  return ExternType{ ExternTypeKind::None, 0 };
}

struct FloatingPointPromotionPattern : mlir::OpRewritePattern<CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(CallOp Call, mlir::PatternRewriter &Rewriter) const override {
    FunctionOp Function = Call.getCalledFunction();
    if (not Function or not Function.isExternal())
      return mlir::failure();

    llvm::StringRef Symbol = Function.getSymName();
    auto I = std::ranges::lower_bound(FloatingPointOperations,
                                      std::string_view(Symbol),
                                      std::less(),
                                      &FloatingPointOperation::Symbol);

    if (I == std::ranges::end(FloatingPointOperations)
        or I->Symbol != std::string_view(Symbol))
      return mlir::failure();

    for (auto [J, T] : llvm::enumerate(Function.getArgumentTypes())) {
      if (J >= std::size(I->ParameterTypes))
        return mlir::failure();

      if (getExternType(T) != I->ParameterTypes[J])
        return mlir::failure();
    }

    mlir::Type ReturnType = Call.getType();
    ExternType ERT = getExternType(ReturnType);

    if (I->ReturnType == Bool) {
      if (ERT.Kind != ExternTypeKind::Unsigned)
        return mlir::failure();

      ReturnType = BoolType::get(Rewriter.getContext());
    } else if (ERT != I->ReturnType) {
      return mlir::failure();
    }

    mlir::OperationState OpState(Call->getLoc(), I->Operation);
    OpState.addTypes(ReturnType);
    OpState.addOperands(Call->getOperands().drop_front());

    Rewriter.setInsertionPoint(Call);
    mlir::Value Value = Rewriter.create(OpState)->getResult(0);

    if (I->ReturnType == Bool)
      Value = Rewriter.create<BoolExtendOp>(Call->getLoc(),
                                            Call.getType(),
                                            Value);

    Rewriter.replaceAllUsesWith(Call, Value);
    Rewriter.eraseOp(Call);

    return mlir::success();
  }
};

struct FloatingPointPromotionPass
  : clift::impl::CliftFloatingPointPromotionBase<FloatingPointPromotionPass> {

  void runOnOperation() override {
    mlir::MLIRContext *Context = &getContext();

    mlir::RewritePatternSet Patterns(Context);
    Patterns.add<FloatingPointPromotionPattern>(Context);

    // TODO: Use walkAndApplyPatterns
    if (mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(Patterns))
          .failed())
      signalPassFailure();
  }
};

} // namespace

PassPtr<FunctionOp> clift::createFloatingPointPromotionPass() {
  return std::make_unique<FloatingPointPromotionPass>();
}
