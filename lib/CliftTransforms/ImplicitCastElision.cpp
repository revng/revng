//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/ScopedExchange.h"
#include "revng/Clift/CliftC.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTIMPLICITCASTELISION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

static bool isBooleanTestedRegion(mlir::Region *R) {
  if (auto For = mlir::dyn_cast<ForOp>(R->getParentOp()))
    return R == &For.getCondition();

  return mlir::isa<IfOp, WhileOp, DoWhileOp>(R->getParentOp());
}

class ImplicitCastElider {
  const CDataModel &DataModel;

  llvm::SmallVector<CastOpInterface> ImplicitConversions;

public:
  static void elide(FunctionOp Function) {
    ImplicitCastElider(getDataModel(Function)).walkAndElide(Function);
  }

private:
  explicit ImplicitCastElider(const CDataModel &DataModel) :
    DataModel(DataModel) {}

  mlir::Type getIntType(mlir::MLIRContext *Context) {
    return IntegerType::get(Context,
                            IntegerKind::Signed,
                            DataModel.getIntSize());
  }

  void addImplicitConversion(CastOpInterface Cast) {
    ImplicitConversions.push_back(Cast);
  }

  bool isVoidToNonVoidPointerCast(CastOpInterface Cast) {
    if (not mlir::isa<BitCastOp>(Cast))
      return false;

    auto TP = clift::unwrapped_dyn_cast<PointerType>(Cast.getType());
    if (not TP or clift::unwrapped_isa<VoidType>(TP.getPointeeType()))
      return false;

    auto SP = clift::unwrapped_dyn_cast<PointerType>(Cast.getValueType());
    return SP and clift::unwrapped_isa<VoidType>(SP.getPointeeType());
  }

  bool isImplicitConversion(CastOpInterface Cast) {
    if (not c::isImplicitConversion(Cast))
      return false;

    // TODO: Add a configuration to enable elision of void* to T* casts.
    if (isVoidToNonVoidPointerCast(Cast))
      return false;

    return true;
  }

  class OperatorType {
  public:
    mlir::Type T1;
    mlir::Type T2;

  public:
    OperatorType() = default;
    OperatorType(mlir::Type T1) : T1(T1), T2() {}
    OperatorType(mlir::Type T1, mlir::Type T2) : T1(T1), T2(T2) {}

    [[nodiscard]] explicit operator bool() const {
      return static_cast<bool>(T1);
    }

    [[nodiscard]] friend bool operator==(const OperatorType &,
                                         const OperatorType &) = default;
  };

  mlir::Type promote(mlir::Type T) {
    auto IntType = getUnderlyingIntegerType(T);
    if (IntType.getSize() < DataModel.getIntSize())
      T = getIntType(T.getContext());

    return T;
  }

  mlir::Type getCommonIntegerType(mlir::MLIRContext *Context,
                                  llvm::ArrayRef<mlir::Type> Types) {
    uint64_t MaxSize = 0;
    uint64_t MaxIsUnsigned = false;

    for (mlir::Type T : Types) {
      auto IntType = getUnderlyingIntegerType(T);
      auto IntSize = IntType.getSize();

      if (IntSize > MaxSize) {
        MaxSize = IntSize;
        MaxIsUnsigned = false;
      }

      if (IntSize == MaxSize)
        MaxSize |= IntType.isUnsigned();
    }

    if (MaxSize < DataModel.getIntSize()) {
      MaxSize = DataModel.getIntSize();
      MaxIsUnsigned = false;
    }

    return IntegerType::get(Context,
                            MaxIsUnsigned ? IntegerKind::Unsigned :
                                            IntegerKind::Signed,
                            MaxSize);
  }

  OperatorType getArithmeticOperatorType(mlir::Operation *Op,
                                         llvm::ArrayRef<mlir::Type> Types) {
    if (mlir::isa<NegOp, BitwiseNotOp>(Op))
      return promote(Types.front());

    if (mlir::isa<ShlOp, ShrOp, SarOp>(Op))
      return { promote(Types.front()), promote(Types.back()) };

    if (mlir::isa<AddOp,
                  SubOp,
                  MulOp,
                  SDivOp,
                  UDivOp,
                  SRemOp,
                  URemOp,
                  BitwiseAndOp,
                  BitwiseOrOp,
                  BitwiseXorOp,
                  CmpEqOp,
                  CmpNeOp,
                  SCmpLtOp,
                  UCmpLtOp,
                  SCmpGtOp,
                  UCmpGtOp,
                  SCmpLeOp,
                  UCmpLeOp,
                  SCmpGeOp,
                  UCmpGeOp>(Op))
      return getCommonIntegerType(Op->getContext(), Types);

    revng_abort();
  }

  void elideBooleanTestingCasts(mlir::OpOperand &Operand) {
    if (auto Test = Operand.get().getDefiningOp<TestOp>())
      addImplicitConversion(Test);
  }

  /// Returns true if the expression is a real boolean expression, potentially
  /// having type bool in C, as opposed to a boolean expression with type int.
  bool isRealBooleanExpression(mlir::Value Value) {
    revng_assert(mlir::isa<BoolType>(Value.getType()));

    mlir::Operation *Op = Value.getDefiningOp();

    // Boolean values always have a defining operation.
    revng_assert(Op != nullptr);

    return mlir::isa<TestOp, TrueOp, FalseOp>(Op);
  }

  void elideBooleanExtensionCasts(BoolExtendOp Extend) {
    if (not isRealBooleanExpression(Extend.getValue())) {
      if (Extend.getType() == getIntType(Extend.getContext()))
        addImplicitConversion(Extend);
    }
  }

  void elideCoercingContextCasts(mlir::OpOperand &Operand) {
    if (auto Cast = Operand.get().getDefiningOp<CastOpInterface>()) {
      if (isImplicitConversion(Cast))
        addImplicitConversion(Cast);
    }
  }

  void elideDecayCasts(mlir::OpOperand &Operand) {
    if (auto Cast = Operand.get().getDefiningOp<DecayOp>()) {
      if (c::isImplicitConversion(Cast))
        addImplicitConversion(Cast);
    }
  }

  void elideAssignmentCasts(AssignOp Op) {
    elideCoercingContextCasts(Op->getOpOperand(1));
  }

  void elideCallCasts(CallOp Op) {
    elideDecayCasts(Op->getOpOperand(0));

    auto FuncType = Op.getFunctionType();
    for (auto [I, T] : llvm::enumerate(FuncType.getArgumentTypes()))
      elideCoercingContextCasts(Op->getOpOperand(I + 1));
  }

  void elideArithmeticCasts(mlir::Operation *Op) {
    llvm::SmallVector<mlir::Type> Types(Op->getOperandTypes());
    auto OperatorType = getArithmeticOperatorType(Op, Types);
    revng_assert(OperatorType);

    for (auto [I, T] : llvm::enumerate(Types)) {
      auto Cast = Op->getOperand(I).getDefiningOp<CastOpInterface>();
      if (not Cast)
        continue;

      mlir::Type CastOperandType = Cast.getValueType();
      if (not unwrapped_isa<IntegralType>(CastOperandType))
        continue;

      ScopedExchange Transaction(T, CastOperandType);
      if (getArithmeticOperatorType(Op, Types) != OperatorType)
        continue;

      Transaction.commit();
      addImplicitConversion(Cast);
    }
  }

  void elideCastsInContext(mlir::Operation *Op) {
    if (Op->hasAttr("clift.intrinsic"))
      return;

    if (auto Yield = mlir::dyn_cast<YieldOp>(Op)) {
      if (isBooleanTestedRegion(Yield->getParentRegion()))
        return elideBooleanTestingCasts(Yield->getOpOperand(0));

      if (mlir::isa<LocalVariableOp, ReturnOp>(Yield->getParentOp()))
        return elideCoercingContextCasts(Yield->getOpOperand(0));
    }

    if (auto Extend = mlir::dyn_cast<BoolExtendOp>(Op))
      return elideBooleanExtensionCasts(Extend);

    if (mlir::isa<TernaryOp, LogicalNotOp>(Op))
      return elideBooleanTestingCasts(Op->getOpOperand(0));

    if (mlir::isa<LogicalAndOp, LogicalOrOp>(Op)) {
      elideBooleanTestingCasts(Op->getOpOperand(0));
      elideBooleanTestingCasts(Op->getOpOperand(1));
      return;
    }

    if (mlir::isa<IndirectionOp, IndirectAccessOp, SubscriptOp>(Op))
      return elideDecayCasts(Op->getOpOperand(0));

    if (auto E = mlir::dyn_cast<AssignOp>(Op))
      return elideAssignmentCasts(E);

    if (auto E = mlir::dyn_cast<CallOp>(Op))
      return elideCallCasts(E);

    if (mlir::isa<NegOp,
                  AddOp,
                  SubOp,
                  MulOp,
                  SDivOp,
                  UDivOp,
                  SRemOp,
                  URemOp,
                  BitwiseNotOp,
                  BitwiseAndOp,
                  BitwiseOrOp,
                  BitwiseXorOp,
                  ShlOp,
                  ShrOp,
                  SarOp>(Op))
      return elideArithmeticCasts(Op);

    if (mlir::isa<CmpEqOp,
                  CmpNeOp,
                  SCmpLtOp,
                  UCmpLtOp,
                  SCmpGtOp,
                  UCmpGtOp,
                  SCmpLeOp,
                  UCmpLeOp,
                  SCmpGeOp,
                  UCmpGeOp>(Op)) {
      if (clift::unwrapped_isa<IntegralType>(Op->getOperand(0).getType()))
        elideArithmeticCasts(Op);
    }
  }

  void walkAndElide(FunctionOp Function) {
    Function->walk([this](mlir::Operation *Op) { elideCastsInContext(Op); });

    if (ImplicitConversions.empty())
      return;

    std::ranges::sort(ImplicitConversions);
    ImplicitConversions.erase(std::unique(ImplicitConversions.begin(),
                                          ImplicitConversions.end()),
                              ImplicitConversions.end());

    mlir::OpBuilder Builder(Function.getContext());
    for (CastOpInterface Cast : ImplicitConversions) {
      mlir::OpOperand *Use = getOnlyUse(Cast);
      revng_assert(Use != nullptr);

      Builder.setInsertionPoint(Cast);
      Use->set(Builder.create<ImplicitCastOp>(Cast->getLoc(),
                                              Cast.getType(),
                                              Cast.getValue()));
      Cast->erase();
    }
  }
};

template<typename T>
using PassBase = clift::impl::CliftImplicitCastElisionBase<T>;

struct ImplicitCastElisionPass : PassBase<ImplicitCastElisionPass> {
  void runOnOperation() override { ImplicitCastElider::elide(getOperation()); }
};

} // namespace

PassPtr<FunctionOp> clift::createImplicitCastElisionPass() {
  return std::make_unique<ImplicitCastElisionPass>();
}
