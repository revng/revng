//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/CliftTransforms/TypeRefinement.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTTYPEREFINEMENT
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

//===--------------------------- Type comparison --------------------------===//

namespace {

// TODO: Use std::bitset when C++23 is available.
template<size_t BitCount>
class Bitset {
  using WordType = size_t;

  static constexpr size_t WordBits = sizeof(WordType) * CHAR_BIT;
  static constexpr size_t WordCount = (BitCount + WordBits - 1) / WordBits;

  WordType Words[WordCount] = {};

public:
  constexpr void set(size_t Index) {
    size_t I = Index / WordBits;
    size_t J = Index % WordBits;
    Words[I] |= static_cast<WordType>(1) << J;
  }

  [[nodiscard]] constexpr bool test(size_t Index) const {
    size_t I = Index / WordBits;
    size_t J = Index % WordBits;
    return Words[I] & static_cast<WordType>(1) << J;
  }
};

enum class TypeKind : uint8_t {
  Void,

  Opaque,
  Generic,
  PointerOrNumber,
  Pointer,
  Number,
  Signed,
  Unsigned,
  FloatingPoint,
  UserDefined,

  Count
};

class TypeTable {
  static constexpr size_t TypeKindCount = static_cast<uint8_t>(TypeKind::Count);
  static constexpr size_t TypeTableSize = TypeKindCount * TypeKindCount;

  Bitset<TypeTableSize> Table = {};

public:
  [[nodiscard]] static bool ordered(TypeKind K1, TypeKind K2);

private:
  consteval TypeTable() {
    // NOTE: Define more refined orderings before less refined ones.
    //
    //       This is simply because of how orderTransitive is defined. If the
    //       less refined ordering is defined first, the transitive propagation
    //       of inherited relations is not applied correctly.

    orderTransitive(TypeKind::Number, TypeKind::Signed);
    orderTransitive(TypeKind::Number, TypeKind::Unsigned);

    orderTransitive(TypeKind::PointerOrNumber, TypeKind::Pointer);
    orderTransitive(TypeKind::PointerOrNumber, TypeKind::Number);

    orderTransitive(TypeKind::Generic, TypeKind::PointerOrNumber);
    orderTransitive(TypeKind::Generic, TypeKind::FloatingPoint);

    orderTransitive(TypeKind::Opaque, TypeKind::Generic);
    orderTransitive(TypeKind::Opaque, TypeKind::UserDefined);
  }

  constexpr bool orderedImpl(TypeKind K1, TypeKind K2) const {
    size_t I = static_cast<size_t>(K1);
    size_t J = static_cast<size_t>(K2);
    return Table.test(I * TypeKindCount + J);
  }

  consteval void orderTransitive(TypeKind K1, TypeKind K2) {
    revng_assert(K1 < K2);

    order(K1, K2);
    for (TypeKind K = TypeKind::Opaque; K < TypeKind::Count; increment(K)) {
      if (orderedImpl(K, K2))
        order(K1, K);
    }
  }

  consteval void order(TypeKind K1, TypeKind K2) {
    size_t I = static_cast<size_t>(K1);
    size_t J = static_cast<size_t>(K2);

    Table.set(I * TypeKindCount + J);
    Table.set(J * TypeKindCount + I);
  }

  static consteval void increment(TypeKind &K) {
    K = static_cast<TypeKind>(static_cast<uint8_t>(K) + 1);
  }
};

bool TypeTable::ordered(TypeKind K1, TypeKind K2) {
  static constexpr TypeTable Table;
  return Table.orderedImpl(K1, K2);
}

static TypeKind getTypeKind(mlir::Type T) {
  revng_assert(not mlir::isa<TypedefType>(T));

  if (mlir::isa<VoidType>(T))
    return TypeKind::Void;

  if (mlir::isa<FloatType>(T))
    return TypeKind::FloatingPoint;

  if (auto U = mlir::dyn_cast<IntegerType>(T)) {
    switch (U.getKind()) {
    case IntegerKind::Generic:
      return TypeKind::Generic;
    case IntegerKind::PointerOrNumber:
      return TypeKind::PointerOrNumber;
    case IntegerKind::Number:
      return TypeKind::Number;
    case IntegerKind::Signed:
      return TypeKind::Signed;
    case IntegerKind::Unsigned:
      return TypeKind::Unsigned;
    }
  }

  if (mlir::isa<PointerType>(T))
    return TypeKind::Pointer;

  if (auto U = mlir::dyn_cast<StructType>(T)) {
    if (U.isOpaque())
      return TypeKind::Opaque;
  }

  return TypeKind::UserDefined;
}

static mlir::Type unwrapType(mlir::Type T) {
  if (auto U = mlir::dyn_cast<EnumType>(T))
    return U.getUnderlyingType();

  if (auto U = mlir::dyn_cast<TypedefType>(T))
    return U.getUnderlyingType();

  return nullptr;
}

/// Returns the fully unwrapped type and number of layers.
static std::pair<mlir::Type, size_t> fullyUnwrapType(mlir::Type T) {
  size_t Count = 0;
  while (auto U = unwrapType(T)) {
    ++Count;
    T = U;
  }
  return { T, Count };
}

static mlir::Type unwrapSomeTypes(mlir::Type T, size_t Count) {
  for (size_t I = 0; I < Count; ++I)
    T = unwrapType(T);
  return T;
}

} // namespace

std::partial_ordering clift::compareTypeRefinement(mlir::Type T1,
                                                   mlir::Type T2) {
  revng_assert(T1);
  revng_assert(T2);

  auto [U1, C1] = fullyUnwrapType(T1);
  auto [U2, C2] = fullyUnwrapType(T2);

  if (equivalent(U1, U2)) {
    if (C1 == 0 or C2 == 0)
      return C1 <=> C2;

    if (C1 > C2) {
      if (equivalent(unwrapSomeTypes(T1, C1 - C2), T2))
        return std::partial_ordering::greater;
    }

    if (C1 < C2) {
      if (equivalent(T1, unwrapSomeTypes(T2, C2 - C1)))
        return std::partial_ordering::less;
    }

    return std::partial_ordering::unordered;
  }

  TypeKind K1 = getTypeKind(U1);
  TypeKind K2 = getTypeKind(U2);

  if (TypeTable::ordered(K1, K2))
    return K1 <=> K2;

  return std::partial_ordering::unordered;
}

//===--------------------------- Type refinement --------------------------===//

namespace {

static mlir::Type getInputType(mlir::Value Value) {
  if (auto Cast = Value.getDefiningOp<BitCastOp>())
    return removeConst(Cast.getValueType());
  return nullptr;
}

static mlir::Type getInputType(mlir::Operation *Op, unsigned Index) {
  return getInputType(Op->getOperand(Index));
}

static std::partial_ordering compareInputRefinement(mlir::Type T1,
                                                    mlir::Type T2) {
  if (not T1)
    return std::partial_ordering::less;

  if (not T2)
    return std::partial_ordering::greater;

  return compareTypeRefinement(T1, T2);
}

static mlir::Value emitTypeConversion(mlir::PatternRewriter &Rewriter,
                                      mlir::Operation *Op,
                                      mlir::Value Value,
                                      mlir::Type NewType) {
  mlir::Location Loc = Op->getLoc();
  if (isLvalueExpression(Value)) {
    const auto &DM = getDataModel(Op->getParentOfType<mlir::ModuleOp>());
    auto P1 = PointerType::get(Value.getType(), DM.PointerSize);
    auto P2 = PointerType::get(NewType, DM.PointerSize);

    Value = Rewriter.create<AddressofOp>(Loc, P1, Value);
    Value = Rewriter.create<BitCastOp>(Loc, P2, Value);
    Value = Rewriter.create<IndirectionOp>(Loc, Value);
  } else {
    Value = Rewriter.create<BitCastOp>(Loc, NewType, Value);
  }
  return Value;
}

static void setOperandType(mlir::PatternRewriter &Rewriter,
                           mlir::Operation *Op,
                           unsigned OperandIndex,
                           mlir::Type NewType) {
  mlir::OpOperand &Operand = Op->getOpOperand(OperandIndex);
  mlir::Type OldType = Operand.get().getType();
  revng_assert(OldType != NewType);

  mlir::OpBuilder::InsertionGuard Guard(Rewriter);
  Rewriter.setInsertionPoint(Op);

  mlir::Value NewValue = emitTypeConversion(Rewriter,
                                            Op,
                                            Operand.get(),
                                            NewType);

  Rewriter.updateRootInPlace(Op, [&]() { Operand.set(NewValue); });
}

static void setResultType(mlir::PatternRewriter &Rewriter,
                          mlir::Operation *Op,
                          mlir::Type NewType) {
  mlir::OpResult Result = Op->getOpResult(0);
  mlir::Type OldType = Result.getType();
  revng_assert(OldType != NewType);

  Rewriter.updateRootInPlace(Op, [&]() { Result.setType(NewType); });

  llvm::SmallVector<mlir::OpOperand *> Operands;
  for (mlir::OpOperand &Operand : Result.getUses()) {
    if (not mlir::isa<RequireOp>(Operand.getOwner()))
      Operands.push_back(&Operand);
  }

  for (mlir::OpOperand *Operand : Operands) {
    mlir::OpBuilder::InsertionGuard Guard(Rewriter);
    Rewriter.setInsertionPoint(Operand->getOwner());

    mlir::Value NewValue = emitTypeConversion(Rewriter, Op, Result, OldType);

    Operand->set(NewValue);
  }
}

static void setInitializerType(mlir::PatternRewriter &Rewriter,
                               LocalVariableOp Local,
                               mlir::Type NewType) {
  if (YieldOp Yield = getYieldOp(Local.getInitializer())) {
    mlir::OpBuilder::InsertionGuard Guard(Rewriter);
    Rewriter.setInsertionPoint(Yield);

    Yield->setOperand(0,
                      Rewriter.create<BitCastOp>(Yield->getLoc(),
                                                 NewType,
                                                 Yield.getValue()));
  }
}

static bool isMoreRefined(mlir::Type LHS, mlir::Type RHS) {
  // For now, arrays not being value types, cannot be refined.
  return not clift::unwrapped_isa<ArrayType>(LHS)
         and not clift::unwrapped_isa<ArrayType>(RHS)
         and compareTypeRefinement(LHS, RHS) > 0;
}

using TypeConstraint = llvm::function_ref<bool(mlir::Type)>;

static bool noConstraint(mlir::Type) {
  return true;
};

template<typename TypeKind>
static bool isaConstraint(mlir::Type T) {
  return clift::unwrapped_isa<TypeKind>(T);
};

struct VariableForwardRefinementPattern : mlir::OpRewritePattern<AssignOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() { addDebugLabels("type-refinement"); }

  mlir::LogicalResult
  matchAndRewrite(AssignOp Assign,
                  mlir::PatternRewriter &Rewriter) const override {
    auto Local = Assign.getLhs().getDefiningOp<LocalVariableOp>();
    if (not Local)
      return mlir::failure();

    mlir::Type NewType = getInputType(Assign.getRhs());
    if (not NewType)
      return mlir::failure();

    if (not isMoreRefined(NewType, Local.getType()))
      return mlir::failure();

    setInitializerType(Rewriter, Local, NewType);
    setResultType(Rewriter, Local, NewType);

    return mlir::success();
  }
};

struct VariableBackwardRefinementPattern : mlir::OpRewritePattern<BitCastOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() { addDebugLabels("type-refinement"); }

  mlir::LogicalResult
  matchAndRewrite(BitCastOp Cast,
                  mlir::PatternRewriter &Rewriter) const override {
    auto Local = Cast.getValue().getDefiningOp<LocalVariableOp>();
    if (not Local)
      return mlir::failure();

    mlir::Type NewType = Cast.getType();

    if (not isMoreRefined(NewType, Local.getType()))
      return mlir::failure();

    setInitializerType(Rewriter, Local, NewType);
    setResultType(Rewriter, Local, NewType);

    return mlir::success();
  }
};

struct ExpressionForwardRefinementPattern
  : mlir::OpInterfaceRewritePattern<ExpressionOpInterface> {

  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  void initialize() { addDebugLabels("type-refinement"); }

  mlir::LogicalResult
  matchAndRewriteUnaryExpression(ExpressionOpInterface Op,
                                 unsigned OperandIndex,
                                 mlir::PatternRewriter &Rewriter,
                                 TypeConstraint Constraint =
                                   noConstraint) const {
    mlir::Type RefinedType = getInputType(Op, OperandIndex);

    if (not RefinedType)
      return mlir::failure();

    if (not Constraint(RefinedType))
      return mlir::failure();

    if (not isMoreRefined(RefinedType, Op.getType()))
      return mlir::failure();

    setOperandType(Rewriter, Op, OperandIndex, RefinedType);
    setResultType(Rewriter, Op, RefinedType);

    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewriteBinaryExpression(ExpressionOpInterface Op,
                                  unsigned OperandIndex1,
                                  unsigned OperandIndex2,
                                  mlir::PatternRewriter &Rewriter,
                                  TypeConstraint Constraint =
                                    noConstraint) const {
    mlir::Type T1 = getInputType(Op, OperandIndex1);
    mlir::Type T2 = getInputType(Op, OperandIndex2);

    if (not T1 and not T2)
      return mlir::failure();

    auto HandleRefinedOperand = [&](mlir::Type RefinedType) {
      if (not Constraint(RefinedType))
        return mlir::failure();

      if (not isMoreRefined(RefinedType, Op.getType()))
        return mlir::failure();

      setOperandType(Rewriter, Op, OperandIndex1, RefinedType);
      setOperandType(Rewriter, Op, OperandIndex2, RefinedType);
      setResultType(Rewriter, Op, RefinedType);

      return mlir::success();
    };

    auto Cmp = compareInputRefinement(T1, T2);

    if (Cmp >= 0)
      return HandleRefinedOperand(T1);

    if (Cmp < 0)
      return HandleRefinedOperand(T2);

    return mlir::failure();
  }

  mlir::LogicalResult
  matchAndRewriteComparison(ExpressionOpInterface Op,
                            mlir::PatternRewriter &Rewriter,
                            TypeConstraint Constraint = noConstraint) const {
    mlir::Type T1 = getInputType(Op, 0);
    mlir::Type T2 = getInputType(Op, 1);

    if (not T1 and not T2)
      return mlir::failure();

    auto HandleRefinedOperand = [&](mlir::Type RefinedType) {
      if (not isScalarType(RefinedType))
        return mlir::failure();

      if (not Constraint(RefinedType))
        return mlir::failure();

      if (not isMoreRefined(RefinedType, Op.getType()))
        return mlir::failure();

      RefinedType = removeConst(RefinedType);
      setOperandType(Rewriter, Op, 0, RefinedType);
      setOperandType(Rewriter, Op, 1, RefinedType);

      return mlir::success();
    };

    auto Cmp = compareInputRefinement(T1, T2);

    if (Cmp >= 0)
      return HandleRefinedOperand(T1);

    if (Cmp < 0)
      return HandleRefinedOperand(T2);

    return mlir::failure();
  }

  mlir::LogicalResult
  matchAndRewrite(ExpressionOpInterface Op,
                  mlir::PatternRewriter &Rewriter) const override {
    if (mlir::isa<NegOp, BitwiseNotOp, ShlOp, ShrOp, SarOp>(Op)) {
      return matchAndRewriteUnaryExpression(Op,
                                            /*OperandIndex=*/0,
                                            Rewriter,
                                            isaConstraint<IntegralType>);
    }

    if (mlir::isa<AddOp,
                  SubOp,
                  MulOp,
                  SDivOp,
                  UDivOp,
                  SRemOp,
                  URemOp,
                  BitwiseAndOp,
                  BitwiseOrOp,
                  BitwiseXorOp>(Op)) {
      return matchAndRewriteBinaryExpression(Op,
                                             /*OperandIndex1=*/0,
                                             /*OperandIndex2=*/1,
                                             Rewriter,
                                             isaConstraint<IntegralType>);
    }

    if (mlir::isa<TernaryOp>(Op))
      return matchAndRewriteBinaryExpression(Op, 1, 2, Rewriter);

    if (mlir::isa<CmpEqOp,
                  CmpNeOp,
                  SCmpLtOp,
                  UCmpLtOp,
                  SCmpGtOp,
                  UCmpGtOp,
                  SCmpLeOp,
                  UCmpLeOp,
                  SCmpGeOp,
                  UCmpGeOp>(Op))
      return matchAndRewriteComparison(Op, Rewriter);

    return mlir::failure();
  }
};

struct ExpressionBackwardRefinementPattern : mlir::OpRewritePattern<BitCastOp> {
  using OpRewritePattern::OpRewritePattern;

  void initialize() { addDebugLabels("type-refinement"); }

  mlir::LogicalResult
  matchAndRewriteUnaryExpression(ExpressionOpInterface Op,
                                 mlir::Type RefinedType,
                                 unsigned OperandIndex,
                                 mlir::PatternRewriter &Rewriter) const {
    if (not isMoreRefined(RefinedType, Op.getType()))
      return mlir::failure();

    setOperandType(Rewriter, Op, OperandIndex, RefinedType);
    setResultType(Rewriter, Op, RefinedType);

    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewriteBinaryExpression(ExpressionOpInterface Op,
                                  mlir::Type RefinedType,
                                  unsigned OperandIndex1,
                                  unsigned OperandIndex2,
                                  mlir::PatternRewriter &Rewriter) const {
    if (not isMoreRefined(RefinedType, Op.getType()))
      return mlir::failure();

    setOperandType(Rewriter, Op, OperandIndex1, RefinedType);
    setOperandType(Rewriter, Op, OperandIndex2, RefinedType);
    setResultType(Rewriter, Op, RefinedType);

    return mlir::success();
  }

  mlir::LogicalResult
  matchAndRewrite(BitCastOp Cast,
                  mlir::PatternRewriter &Rewriter) const override {
    auto Op = Cast.getValue().getDefiningOp<ExpressionOpInterface>();
    if (not Op)
      return mlir::failure();

    mlir::Type NewType = Cast.getType();

    // For now, arrays not being value types, cannot be refined.
    if (clift::unwrapped_isa<ArrayType>(NewType))
      return mlir::failure();

    if (mlir::isa<NegOp, BitwiseNotOp, ShlOp, ShrOp, SarOp>(Op)) {
      if (not clift::unwrapped_isa<IntegralType>(NewType))
        return mlir::failure();

      return matchAndRewriteUnaryExpression(Op, NewType, 0, Rewriter);
    }

    if (mlir::isa<AddOp,
                  SubOp,
                  MulOp,
                  SDivOp,
                  UDivOp,
                  SRemOp,
                  URemOp,
                  BitwiseAndOp,
                  BitwiseOrOp,
                  BitwiseXorOp>(Op)) {
      if (not clift::unwrapped_isa<IntegralType>(NewType))
        return mlir::failure();

      return matchAndRewriteBinaryExpression(Op, NewType, 0, 1, Rewriter);
    }

    if (mlir::isa<TernaryOp>(Op))
      return matchAndRewriteBinaryExpression(Op, NewType, 1, 2, Rewriter);

    return mlir::failure();
  }
};

struct TypeRefinementPass
  : clift::impl::CliftTypeRefinementBase<TypeRefinementPass> {

  mlir::FrozenRewritePatternSet Patterns;

  mlir::LogicalResult initialize(mlir::MLIRContext *Context) override {
    mlir::RewritePatternSet Set(Context);

    populateWithTypeRefinementPatterns(Set);
    Patterns = mlir::FrozenRewritePatternSet(std::move(Set));

    return mlir::success();
  }

  void runOnOperation() override {
    FunctionOp Function = getOperation();

    mlir::GreedyRewriteConfig Config;
    Config.maxIterations = mlir::GreedyRewriteConfig::kNoLimit;
    if (mlir::applyPatternsAndFoldGreedily(Function, Patterns, Config).failed())
      signalPassFailure();
  }
};

} // namespace

void clift::populateWithTypeRefinementPatterns(mlir::RewritePatternSet &Set) {
  Set.add<VariableForwardRefinementPattern>(Set.getContext());
  Set.add<VariableBackwardRefinementPattern>(Set.getContext());

  Set.add<ExpressionForwardRefinementPattern>(Set.getContext());
  Set.add<ExpressionBackwardRefinementPattern>(Set.getContext());
}

PassPtr<FunctionOp> clift::createTypeRefinementPass() {
  return std::make_unique<TypeRefinementPass>();
}
