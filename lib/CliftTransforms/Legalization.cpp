//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Legalization.h"
#include "revng/CliftTransforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTCLEGALIZATION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;
using namespace clift;

namespace {

static IntegerType getIntType(mlir::MLIRContext *Context,
                              const TargetCImplementation &Target) {
  return IntegerType::get(Context, IntegerKind::Signed, Target.getIntSize());
}

struct TargetInfo {
  const TargetCImplementation &Target;
  IntegerType IntType;

  explicit TargetInfo(mlir::MLIRContext *Context,
                      const TargetCImplementation &Target) :
    Target(Target), IntType(getIntType(Context, Target)) {}
};

static mlir::OpOperand &getOnlyUse(mlir::Value Value) {
  revng_assert(Value.hasOneUse());
  return *Value.use_begin();
}

template<typename ResizeCastOpOrVoid = void>
static mlir::Value emitCast(mlir::PatternRewriter &Rewriter,
                            mlir::Location Loc,
                            mlir::Value Value,
                            mlir::Type NewType) {
  mlir::Type OldType = Value.getType();

  uint64_t OldSize = getObjectSize(OldType);
  uint64_t NewSize = getObjectSize(NewType);

  if (OldSize == NewSize)
    return Rewriter.create<BitCastOp>(Loc, NewType, Value);

  if constexpr (std::is_void_v<ResizeCastOpOrVoid>) {
    if (NewSize > OldSize)
      return Rewriter.create<ExtendOp>(Loc, NewType, Value);
    else
      return Rewriter.create<TruncateOp>(Loc, NewType, Value);
  } else {
    return Rewriter.create<ResizeCastOpOrVoid>(Loc, NewType, Value);
  }
}

/// Changes the type of the first result of the expression \p Op to \p NewType.
///
/// If \p PreserveExpressionType is true and the result is not discarded, a
/// truncating or extending cast (depending on relative sizes of the two types)
/// is inserted between \p Op and its user. The caller may set this to false
/// when it is known that the change in type has no effect on the semantics of
/// the user of the result.
template<typename ResizeCastOpOrVoid = void>
static void modifyResultType(mlir::PatternRewriter &Rewriter,
                             mlir::Operation *Op,
                             mlir::Type NewType,
                             bool PreserveExpressionType = true) {
  mlir::OpResult Result = Op->getOpResult(0);
  mlir::OpOperand &OnlyUse = getOnlyUse(Result);

  mlir::Type OldType = Result.getType();
  Result.setType(NewType);

  if (PreserveExpressionType and not clift::isDiscarded(Result)) {
    Rewriter.setInsertionPointAfter(Op);
    OnlyUse.set(emitCast<ResizeCastOpOrVoid>(Rewriter,
                                             Op->getLoc(),
                                             Result,
                                             OldType));
  }
}

template<typename ResizeCastOpOrVoid = void>
static void modifyOperandType(mlir::PatternRewriter &Rewriter,
                              mlir::OpOperand &Operand,
                              mlir::Type NewType) {
  mlir::Operation *Op = Operand.getOwner();
  mlir::Value Value = Operand.get();

  Rewriter.setInsertionPoint(Op);
  Operand
    .set(emitCast<ResizeCastOpOrVoid>(Rewriter, Op->getLoc(), Value, NewType));
}

template<typename OpT>
struct PointerResizePattern : mlir::OpRewritePattern<OpT> {
  uint64_t TargetPointerSize;

  explicit PointerResizePattern(mlir::MLIRContext *Context,
                                const TargetInfo &Target) :
    mlir::OpRewritePattern<OpT>(Context),
    TargetPointerSize(Target.Target.PointerSize) {}

  clift::PointerType
  makeTargetPointerType(clift::PointerType OldPointerType) const {
    return clift::PointerType::get(OldPointerType.getPointeeType(),
                                   TargetPointerSize);
  }

  clift::IntegerType
  makeTargetIntegerType(mlir::PatternRewriter &Rewriter,
                        clift::IntegerType OldIntegerType) const {
    return clift::IntegerType::get(Rewriter.getContext(),
                                   OldIntegerType.getKind(),
                                   TargetPointerSize);
  }

  mlir::LogicalResult replacePointerOperand(mlir::PatternRewriter &Rewriter,
                                            clift::ExpressionOpInterface Op,
                                            unsigned Index = 0) const {
    mlir::OpOperand &Operand = Op->getOpOperand(Index);

    auto OldType = clift::unwrapped_dyn_cast<PointerType>(Operand.get()
                                                            .getType());
    if (not OldType or OldType.getPointerSize() == TargetPointerSize)
      return mlir::failure();

    auto NewType = makeTargetPointerType(OldType);
    modifyOperandType<PtrResizeOp>(Rewriter, Operand, NewType);

    return mlir::success();
  }

  mlir::LogicalResult replaceIntegerOperand(mlir::PatternRewriter &Rewriter,
                                            clift::ExpressionOpInterface Op,
                                            unsigned Index = 0) const {
    mlir::OpOperand &Operand = Op->getOpOperand(Index);

    auto OldType = clift::unwrapped_dyn_cast<IntegerType>(Operand.get()
                                                            .getType());
    if (not OldType or OldType.getSize() == TargetPointerSize)
      return mlir::failure();

    auto NewType = makeTargetIntegerType(Rewriter, OldType);
    modifyOperandType(Rewriter, Operand, NewType);

    return mlir::success();
  }

  mlir::LogicalResult
  replacePointerResult(mlir::PatternRewriter &Rewriter,
                       clift::ExpressionOpInterface Op) const {
    auto OldType = clift::unwrapped_cast<PointerType>(Op->getResult(0)
                                                        .getType());

    if (OldType.getPointerSize() == TargetPointerSize)
      return mlir::failure();

    auto NewType = makeTargetPointerType(OldType);
    modifyResultType<PtrResizeOp>(Rewriter, Op, NewType);

    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(OpT Op, mlir::PatternRewriter &Rewriter) const override {
    return replacePointerOperand(Rewriter, Op);
  }
};

template<typename OpT>
struct ResizePointerArithmeticPattern : PointerResizePattern<OpT> {
  using PointerResizePattern<OpT>::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(OpT Op, mlir::PatternRewriter &Rewriter) const override {
    unsigned Index = Op.getPointerOperandIndex();
    if (this->replacePointerOperand(Rewriter, Op, Index).failed())
      return mlir::failure();

    auto R1 = this->replaceIntegerOperand(Rewriter, Op, Index ^ 1);
    revng_assert(R1.succeeded());

    auto R2 = this->replacePointerResult(Rewriter, Op);
    revng_assert(R2.succeeded());

    return mlir::success();
  }
};

using ResizePtrAddPattern = ResizePointerArithmeticPattern<clift::PtrAddOp>;
using ResizePtrSubPattern = ResizePointerArithmeticPattern<clift::PtrSubOp>;

struct ResizePtrDiffPattern : PointerResizePattern<clift::PtrDiffOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult replaceIntegerResult(mlir::PatternRewriter &Rewriter,
                                           clift::PtrDiffOp Op) const {
    auto OldType = clift::unwrapped_cast<IntegerType>(Op->getResult(0)
                                                        .getType());

    if (OldType.getSize() == TargetPointerSize)
      return mlir::failure();

    auto NewType = makeTargetIntegerType(Rewriter, OldType);
    modifyResultType(Rewriter, Op, NewType);

    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(clift::PtrDiffOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    if (replacePointerOperand(Rewriter, Op, 0).failed())
      return mlir::failure();

    auto R1 = replacePointerOperand(Rewriter, Op, 1);
    revng_assert(R1.succeeded());

    auto R2 = replaceIntegerResult(Rewriter, Op);
    revng_assert(R2.succeeded());

    return mlir::success();
  }
};

struct ResizeAddressofPattern : PointerResizePattern<clift::AddressofOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::AddressofOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    return replacePointerResult(Rewriter, Op);
  }
};

struct ResizeDecayCastPattern : PointerResizePattern<clift::DecayOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::DecayOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    return replacePointerResult(Rewriter, Op);
  }
};

struct BooleanCanonicalizationPattern
  : mlir::OpTraitRewritePattern<mlir::OpTrait::clift::ReturnsBoolean> {
  IntegerType IntType;

  explicit BooleanCanonicalizationPattern(mlir::MLIRContext *Context,
                                          const TargetInfo &Target) :
    mlir::OpTraitRewritePattern<mlir::OpTrait::clift::ReturnsBoolean>(Context),
    IntType(Target.IntType) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Value Result = Op->getResult(0);

    auto T = clift::unwrapped_cast<IntegerType>(Result.getType());

    if (T.getSize() == IntType.getSize())
      return mlir::failure();

    modifyResultType(Rewriter, Op, IntType, not clift::isBooleanTested(Result));

    return mlir::success();
  }
};

template<typename OpT>
struct ArithmeticPromotionPattern : mlir::OpRewritePattern<OpT> {
  IntegerType IntType;

  explicit ArithmeticPromotionPattern(mlir::MLIRContext *Context,
                                      const TargetInfo &Target) :
    mlir::OpRewritePattern<OpT>(Context), IntType(Target.IntType) {}

  mlir::LogicalResult tryPromoteTypes(mlir::PatternRewriter &Rewriter,
                                      clift::ExpressionOpInterface Op,
                                      llvm::ArrayRef<unsigned> Indices) const {
    mlir::OpResult Result = Op->getOpResult(0);

    auto OldType = clift::getUnderlyingIntegerType(Result.getType());
    if (not OldType or OldType.getSize() >= IntType.getSize())
      return mlir::failure();

    modifyResultType(Rewriter, Op, IntType);

    for (unsigned Index : Indices) {
      mlir::OpOperand &Operand = Op->getOpOperand(Index);
      revng_assert(Operand.get().getType() == OldType);
      modifyOperandType(Rewriter, Operand, IntType);
    }

    return mlir::success();
  }

  mlir::LogicalResult tryPromoteTypes(mlir::PatternRewriter &Rewriter,
                                      clift::ExpressionOpInterface Op) const {
    unsigned Indices[] = { 0, 1 };
    return tryPromoteTypes(Rewriter,
                           Op,
                           llvm::ArrayRef(Indices)
                             .take_front(Op->getNumOperands()));
  }

  mlir::LogicalResult
  matchAndRewrite(OpT Op, mlir::PatternRewriter &Rewriter) const override {
    return tryPromoteTypes(Rewriter, Op);
  }
};

template<typename OpT>
struct ShiftPromotionPattern : ArithmeticPromotionPattern<OpT> {
  using ArithmeticPromotionPattern<OpT>::ArithmeticPromotionPattern;

  mlir::LogicalResult
  matchAndRewrite(OpT Op, mlir::PatternRewriter &Rewriter) const override {
    return this->tryPromoteTypes(Rewriter, Op, { 0 });
  }
};

// Introduces casts around immediates not directly representable in C:
// * 0 -> (int16_t)0, where the original expression has type int16_t.
// * 0 -> (int64_t)0, where the original expression has extended integer type.
// * 0 -> (my_enum)0, where the original expression has type my_enum and my_enum
//                    does not have an enumerator with a value of 0.
struct ImmediateCastPattern : mlir::OpRewritePattern<ImmediateOp> {
  const TargetCImplementation &Target;
  uint64_t IntSize;

  explicit ImmediateCastPattern(mlir::MLIRContext *Context,
                                const TargetInfo &Target) :
    mlir::OpRewritePattern<ImmediateOp>(Context),
    Target(Target.Target),
    IntSize(Target.IntType.getSize()) {}

  mlir::LogicalResult rewriteWithCast(ImmediateOp Op,
                                      mlir::Type NewImmediateType,
                                      mlir::PatternRewriter &Rewriter) const {
    mlir::Value Result = Op.getResult();
    mlir::Type OldImmediateType = Result.getType();
    mlir::OpOperand &Use = getOnlyUse(Result);

    Rewriter.setInsertionPointAfter(Op);
    Result.setType(NewImmediateType);
    Use.set(emitCast(Rewriter, Op->getLoc(), Result, OldImmediateType));

    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewriteEnumImmediate(ImmediateOp Op,
                               EnumType Type,
                               mlir::PatternRewriter &Rewriter) const {
    auto Enumerator = Type.getFieldByValue(Op.getValue());
    if (Enumerator)
      return mlir::failure();

    return rewriteWithCast(Op, Type.getUnderlyingType(), Rewriter);
  }

  bool isRepresentableLiteralSize(uint64_t Size) const {
    auto C = Target.getIntegerKind(Size);
    return C and CIntegerKind::Int <= *C and *C <= CIntegerKind::LongLong;
  }

  mlir::LogicalResult
  matchAndRewriteIntegerImmediate(ImmediateOp Op,
                                  IntegerType Type,
                                  mlir::PatternRewriter &Rewriter) const {
    if (isRepresentableLiteralSize(Type.getSize()))
      return mlir::failure();

    // Sizes in the range [sizeof(int), 8] must be representable in the target.
    uint64_t NewSize = std::clamp<uint64_t>(Type.getSize(), IntSize, 8);

    revng_assert(NewSize != Type.getSize());
    revng_assert(isRepresentableLiteralSize(NewSize));

    IntegerType NewType = Type.getSize() == NewSize ?
                            Type :
                            IntegerType::get(Type.getContext(),
                                             Type.getKind(),
                                             NewSize);

    return rewriteWithCast(Op, NewType, Rewriter);
  }

  mlir::LogicalResult
  matchAndRewrite(ImmediateOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Type Type = unwrapTypedefs(Op.getResult().getType());

    if (auto T = mlir::dyn_cast<EnumType>(Type))
      return matchAndRewriteEnumImmediate(Op, T, Rewriter);

    if (auto T = mlir::dyn_cast<IntegerType>(Type))
      return matchAndRewriteIntegerImmediate(Op, T, Rewriter);

    return mlir::failure();
  }
};

struct CLegalizationPass
  : clift::impl::CliftCLegalizationBase<CLegalizationPass> {

  const TargetCImplementation &Target;

  explicit CLegalizationPass(const TargetCImplementation &Target) :
    Target(Target) {}

  void runOnOperation() override {
    if (legalizeForC(getOperation(), Target).failed())
      signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult clift::legalizeForC(clift::FunctionOp Function,
                                        const TargetCImplementation &Target) {
  mlir::MLIRContext *Context = Function.getContext();
  mlir::RewritePatternSet Set(Context);

  TargetInfo T(Context, Target);

  // Pointer resizing
  Set.add<ResizePtrAddPattern>(Context, T);
  Set.add<ResizePtrSubPattern>(Context, T);
  Set.add<ResizePtrDiffPattern>(Context, T);
  Set.add<PointerResizePattern<IndirectionOp>>(Context, T);
  Set.add<PointerResizePattern<SubscriptOp>>(Context, T);
  Set.add<PointerResizePattern<AccessOp>>(Context, T);
  Set.add<PointerResizePattern<CallOp>>(Context, T);
  Set.add<ResizeAddressofPattern>(Context, T);
  Set.add<ResizeDecayCastPattern>(Context, T);

  // Boolean canonicalization
  Set.add<BooleanCanonicalizationPattern>(Context, T);

  // Integer promotion
  Set.add<ArithmeticPromotionPattern<NegOp>>(Context, T);
  Set.add<ArithmeticPromotionPattern<AddOp>>(Context, T);
  Set.add<ArithmeticPromotionPattern<SubOp>>(Context, T);
  Set.add<ArithmeticPromotionPattern<MulOp>>(Context, T);
  Set.add<ArithmeticPromotionPattern<DivOp>>(Context, T);
  Set.add<ArithmeticPromotionPattern<RemOp>>(Context, T);
  Set.add<ArithmeticPromotionPattern<BitwiseNotOp>>(Context, T);
  Set.add<ArithmeticPromotionPattern<BitwiseAndOp>>(Context, T);
  Set.add<ArithmeticPromotionPattern<BitwiseOrOp>>(Context, T);
  Set.add<ArithmeticPromotionPattern<BitwiseXorOp>>(Context, T);
  Set.add<ShiftPromotionPattern<ShiftLeftOp>>(Context, T);
  Set.add<ShiftPromotionPattern<ShiftRightOp>>(Context, T);

  // Literal typing
  Set.add<ImmediateCastPattern>(Context, T);

  auto Patterns = mlir::FrozenRewritePatternSet(std::move(Set));
  return mlir::applyPatternsAndFoldGreedily(Function, Patterns);
}

clift::PassPtr<clift::FunctionOp>
clift::createCLegalizationPass(const TargetCImplementation &Target) {
  return std::make_unique<CLegalizationPass>(Target);
}
