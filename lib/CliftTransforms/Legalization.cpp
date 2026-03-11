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

static mlir::OpOperand &getOnlyUse(mlir::Value Value) {
  revng_assert(Value.hasOneUse());
  return *Value.use_begin();
}

template<typename ResizeCastOpOrVoid>
static mlir::Value emitCast(mlir::PatternRewriter &Rewriter,
                            mlir::Location Loc,
                            mlir::Value Value,
                            clift::ValueType NewType) {
  clift::ValueType OldType = Value.getType();

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
                             clift::ValueType NewType,
                             bool PreserveExpressionType = true) {
  mlir::OpResult Result = Op->getOpResult(0);
  mlir::OpOperand &OnlyUse = getOnlyUse(Result);

  auto OldType = mlir::cast<clift::ValueType>(Result.getType());
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
                              clift::ValueType NewType) {
  mlir::Operation *Op = Operand.getOwner();
  mlir::Value Value = Operand.get();

  Rewriter.setInsertionPoint(Op);
  Operand
    .set(emitCast<ResizeCastOpOrVoid>(Rewriter, Op->getLoc(), Value, NewType));
}

template<typename OpT>
struct PointerResizePattern : mlir::OpRewritePattern<OpT> {
  explicit PointerResizePattern(mlir::MLIRContext *Context,
                                const TargetCImplementation &Target) :
    mlir::OpRewritePattern<OpT>(Context),
    TargetPointerSize(Target.PointerSize) {}

  uint64_t TargetPointerSize;

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

    auto OldType = clift::getPointerType(Operand.get().getType());
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

    auto OldType = clift::getPrimitiveIntegerType(Operand.get().getType());
    if (not OldType or OldType.getSize() == TargetPointerSize)
      return mlir::failure();

    auto NewType = makeTargetIntegerType(Rewriter, OldType);
    modifyOperandType(Rewriter, Operand, NewType);

    return mlir::success();
  }

  mlir::LogicalResult
  replacePointerResult(mlir::PatternRewriter &Rewriter,
                       clift::ExpressionOpInterface Op) const {
    auto OldType = clift::getPointerType(Op->getResult(0).getType());
    revng_assert(OldType);

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
    auto OldType = clift::getPrimitiveIntegerType(Op->getResult(0).getType());
    revng_assert(OldType);

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

  explicit BooleanCanonicalizationPattern(mlir::MLIRContext *Context,
                                          const TargetCImplementation &Target) :
    mlir::OpTraitRewritePattern<mlir::OpTrait::clift::ReturnsBoolean>(Context),
    CanonicalBooleanType(getCanonicalBooleanType(Context, Target)) {}

  clift::IntegerType CanonicalBooleanType;

  static clift::IntegerType
  getCanonicalBooleanType(mlir::MLIRContext *Context,
                          const TargetCImplementation &Target) {
    return clift::IntegerType::get(Context,
                                   clift::IntegerKind::Signed,
                                   Target.getIntSize(),
                                   /*Const=*/false);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Value Result = Op->getResult(0);

    auto T = mlir::dyn_cast<IntegerType>(unwrapTypedefs(Result.getType()));

    if (T.getSize() == CanonicalBooleanType.getSize())
      return mlir::failure();

    modifyResultType(Rewriter,
                     Op,
                     CanonicalBooleanType,
                     not clift::isBooleanTested(Result));

    return mlir::success();
  }
};

/// If TreatAsBoolean is true, the expression type is not preserved in
/// boolean-tested contexts. See modifyResultType documentation above.
template<typename OpT, bool TreatAsBoolean = false>
struct IntegerPromotionPattern : mlir::OpRewritePattern<OpT> {
  explicit IntegerPromotionPattern(mlir::MLIRContext *Context,
                                   const TargetCImplementation &Target) :
    mlir::OpRewritePattern<OpT>(Context), PromotionSize(Target.getIntSize()) {}

  uint64_t PromotionSize;

  clift::IntegerType makePromotedType(clift::IntegerType Type) const {
    return clift::IntegerType::get(Type.getContext(),
                                   Type.getKind(),
                                   PromotionSize,
                                   /*Const=*/false);
  }

  mlir::LogicalResult tryPromoteTypes(mlir::PatternRewriter &Rewriter,
                                      clift::ExpressionOpInterface Op,
                                      llvm::ArrayRef<unsigned> Indices) const {
    mlir::OpResult Result = Op->getOpResult(0);

    auto OldType = clift::getUnderlyingIntegerType(Result.getType());
    if (not OldType or OldType.getSize() >= PromotionSize)
      return mlir::failure();

    auto NewType = makePromotedType(OldType);
    modifyResultType(Rewriter,
                     Op,
                     NewType,
                     not TreatAsBoolean or not clift::isBooleanTested(Result));

    for (unsigned Index : Indices) {
      mlir::OpOperand &Operand = Op->getOpOperand(Index);
      revng_assert(Operand.get().getType() == OldType);
      modifyOperandType(Rewriter, Operand, NewType);
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
struct ShiftPromotionPattern : IntegerPromotionPattern<OpT> {
  using IntegerPromotionPattern<OpT>::IntegerPromotionPattern;

  mlir::LogicalResult
  matchAndRewrite(OpT Op, mlir::PatternRewriter &Rewriter) const override {
    return this->tryPromoteTypes(Rewriter, Op, { 0 });
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

  // Pointer resizing
  Set.add<ResizePtrAddPattern>(Context, Target);
  Set.add<ResizePtrSubPattern>(Context, Target);
  Set.add<ResizePtrDiffPattern>(Context, Target);
  Set.add<PointerResizePattern<IndirectionOp>>(Context, Target);
  Set.add<PointerResizePattern<SubscriptOp>>(Context, Target);
  Set.add<PointerResizePattern<AccessOp>>(Context, Target);
  Set.add<PointerResizePattern<CallOp>>(Context, Target);
  Set.add<ResizeAddressofPattern>(Context, Target);
  Set.add<ResizeDecayCastPattern>(Context, Target);

  // Boolean canonicalization
  Set.add<BooleanCanonicalizationPattern>(Context, Target);

  // Integer promotion
  Set.add<IntegerPromotionPattern<ImmediateOp,
                                  /*TreatAsBoolean=*/true>>(Context, Target);
  Set.add<IntegerPromotionPattern<NegOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<AddOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<SubOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<MulOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<DivOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<RemOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<BitwiseNotOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<BitwiseAndOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<BitwiseOrOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<BitwiseXorOp>>(Context, Target);
  Set.add<ShiftPromotionPattern<ShiftLeftOp>>(Context, Target);
  Set.add<ShiftPromotionPattern<ShiftRightOp>>(Context, Target);

  auto Patterns = mlir::FrozenRewritePatternSet(std::move(Set));
  return mlir::applyPatternsAndFoldGreedily(Function, Patterns);
}

clift::PassPtr<clift::FunctionOp>
clift::createCLegalizationPass(const TargetCImplementation &Target) {
  return std::make_unique<CLegalizationPass>(Target);
}
