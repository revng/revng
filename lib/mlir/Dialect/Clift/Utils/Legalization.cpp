//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/mlir/Dialect/Clift/Utils/Legalization.h"

namespace clift = mlir::clift;

namespace {

static mlir::OpOperand &getOnlyUse(mlir::Value Value) {
  revng_assert(Value.hasOneUse());
  return *Value.use_begin();
}

static void modifyResultType(mlir::PatternRewriter &Rewriter,
                             mlir::Operation *Op,
                             clift::ValueType NewType) {
  mlir::OpResult Result = Op->getOpResult(0);
  mlir::OpOperand &OnlyUse = getOnlyUse(Result);

  auto OldType = mlir::cast<clift::ValueType>(Result.getType());
  auto CastKind = OldType.getByteSize() > NewType.getByteSize() ?
                    clift::CastKind::Extend :
                    clift::CastKind::Truncate;

  Result.setType(NewType);

  Rewriter.setInsertionPointAfter(Op);
  OnlyUse.set(Rewriter.create<clift::CastOp>(Op->getLoc(),
                                             OldType,
                                             Result,
                                             CastKind));
}

static void modifyOperandType(mlir::PatternRewriter &Rewriter,
                              mlir::OpOperand &Operand,
                              clift::ValueType NewType) {
  mlir::Operation *Op = Operand.getOwner();
  mlir::Value Value = Operand.get();

  auto OldType = mlir::cast<clift::ValueType>(Value.getType());
  auto CastKind = OldType.getByteSize() < NewType.getByteSize() ?
                    clift::CastKind::Extend :
                    clift::CastKind::Truncate;

  Rewriter.setInsertionPoint(Op);
  Operand.set(Rewriter.create<clift::CastOp>(Op->getLoc(),
                                             NewType,
                                             Value,
                                             CastKind));
}

template<typename OpT>
struct PointerResizePattern : mlir::OpRewritePattern<OpT> {
  explicit PointerResizePattern(mlir::MLIRContext *Context,
                                const clift::TargetCImplementation &Target) :
    mlir::OpRewritePattern<OpT>(Context),
    TargetPointerSize(Target.PointerSize) {}

  uint64_t TargetPointerSize;

  clift::PointerType
  makeTargetPointerType(clift::PointerType OldPointerType) const {
    return clift::PointerType::get(OldPointerType.getPointeeType(),
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
    modifyOperandType(Rewriter, Operand, NewType);

    return mlir::success();
  }

  mlir::LogicalResult
  replacePointerResult(mlir::PatternRewriter &Rewriter,
                       clift::ExpressionOpInterface Op) const {
    auto Result = Op->getResult(0);

    auto OldType = clift::getPointerType(Result.getType());
    revng_assert(OldType);

    if (OldType.getPointerSize() == TargetPointerSize)
      return mlir::failure();

    auto NewType = makeTargetPointerType(OldType);
    modifyResultType(Rewriter, Op, NewType);

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

    auto R = this->replacePointerResult(Rewriter, Op);
    revng_assert(R.succeeded());

    return mlir::success();
  }
};

using ResizePtrAddPattern = ResizePointerArithmeticPattern<clift::PtrAddOp>;
using ResizePtrSubPattern = ResizePointerArithmeticPattern<clift::PtrSubOp>;

struct ResizePtrDiffPattern : PointerResizePattern<clift::PtrDiffOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::PtrDiffOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    if (replacePointerOperand(Rewriter, Op, 0).failed())
      return mlir::failure();

    auto R = replacePointerOperand(Rewriter, Op, 1);
    revng_assert(R.succeeded());

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

struct ResizeDecayCastPattern : PointerResizePattern<clift::CastOp> {
  using PointerResizePattern::PointerResizePattern;

  mlir::LogicalResult
  matchAndRewrite(clift::CastOp Op,
                  mlir::PatternRewriter &Rewriter) const override {
    if (Op.getKind() != clift::CastKind::Decay)
      return mlir::failure();

    return replacePointerResult(Rewriter, Op);
  }
};

struct BooleanCanonicalizationPattern
  : mlir::OpTraitRewritePattern<mlir::OpTrait::clift::ReturnsBoolean> {

  explicit BooleanCanonicalizationPattern(mlir::MLIRContext *Context,
                                          const clift::TargetCImplementation
                                            &Target) :
    mlir::OpTraitRewritePattern<mlir::OpTrait::clift::ReturnsBoolean>(Context),
    CanonicalBooleanType(getCanonicalBooleanType(Context, Target)) {}

  clift::PrimitiveType CanonicalBooleanType;

  static clift::PrimitiveType
  getCanonicalBooleanType(mlir::MLIRContext *Context,
                          const clift::TargetCImplementation &Target) {
    return clift::PrimitiveType::get(Context,
                                     clift::PrimitiveKind::SignedKind,
                                     Target.getIntSize(),
                                     /*Const=*/false);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *Op,
                  mlir::PatternRewriter &Rewriter) const override {
    mlir::Value Result = Op->getResult(0);

    auto T = mlir::dyn_cast<clift::PrimitiveType>(clift::dealias(Result
                                                                   .getType(),
                                                                 true));
    revng_assert(T and isIntegerKind(T.getKind()));

    if (T.getSize() == CanonicalBooleanType.getSize())
      return mlir::failure();

    modifyResultType(Rewriter, Op, CanonicalBooleanType);
    return mlir::success();
  }
};

template<typename OpT>
struct IntegerPromotionPattern : mlir::OpRewritePattern<OpT> {

  explicit IntegerPromotionPattern(mlir::MLIRContext *Context,
                                   const clift::TargetCImplementation &Target) :
    mlir::OpRewritePattern<OpT>(Context), PromotionSize(Target.getIntSize()) {}

  uint64_t PromotionSize;

  clift::PrimitiveType makePromotedType(clift::PrimitiveType Type) const {
    return clift::PrimitiveType::get(Type.getContext(),
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
    modifyResultType(Rewriter, Op, NewType);

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

} // namespace

mlir::LogicalResult
clift::legalizeForC(clift::FunctionOp Function,
                    const clift::TargetCImplementation &Target) {
  mlir::MLIRContext *Context = Function.getContext();
  mlir::RewritePatternSet Set(Context);

  // Pointer resizing
  Set.add<ResizePtrAddPattern>(Context, Target);
  Set.add<ResizePtrSubPattern>(Context, Target);
  Set.add<ResizePtrDiffPattern>(Context, Target);
  Set.add<PointerResizePattern<clift::IndirectionOp>>(Context, Target);
  Set.add<PointerResizePattern<clift::SubscriptOp>>(Context, Target);
  Set.add<PointerResizePattern<clift::AccessOp>>(Context, Target);
  Set.add<PointerResizePattern<clift::CallOp>>(Context, Target);
  Set.add<ResizeAddressofPattern>(Context, Target);
  Set.add<ResizeDecayCastPattern>(Context, Target);

  // Boolean canonicalization
  Set.add<BooleanCanonicalizationPattern>(Context, Target);

  // Integer promotion
  Set.add<IntegerPromotionPattern<clift::ImmediateOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<clift::NegOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<clift::AddOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<clift::SubOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<clift::MulOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<clift::DivOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<clift::RemOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<clift::BitwiseNotOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<clift::BitwiseAndOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<clift::BitwiseOrOp>>(Context, Target);
  Set.add<IntegerPromotionPattern<clift::BitwiseXorOp>>(Context, Target);
  Set.add<ShiftPromotionPattern<clift::ShiftLeftOp>>(Context, Target);
  Set.add<ShiftPromotionPattern<clift::ShiftRightOp>>(Context, Target);

  auto Patterns = mlir::FrozenRewritePatternSet(std::move(Set));
  return mlir::applyPatternsAndFoldGreedily(Function, Patterns);
}
