//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/ScopedExchange.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTIMPLICITCASTELISION
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

using namespace clift;

namespace {

class ImplicitCastElider {
  const CDataModel &DataModel;

public:
  explicit ImplicitCastElider(const CDataModel &DataModel) :
    DataModel(DataModel) {}

  void elide(mlir::Operation *Op) {
    if (auto Yield = mlir::dyn_cast<YieldOp>(Op)) {
      if (mlir::isa<LocalVariableOp, ReturnOp>(Yield->getParentOp()))
        return elideCoercingContextCasts(Yield->getOpOperand(0));
    }

    if (auto Assignment = mlir::dyn_cast<AssignOp>(Op))
      return elideAssignmentCasts(Assignment);

    if (auto Call = mlir::dyn_cast<CallOp>(Op))
      return elideArgumentCasts(Call);

    if (mlir::isa<NegOp,
                  AddOp,
                  SubOp,
                  MulOp,
                  DivOp,
                  RemOp,
                  BitwiseNotOp,
                  BitwiseAndOp,
                  BitwiseOrOp,
                  BitwiseXorOp,
                  ShiftLeftOp,
                  ShiftRightOp,
                  CmpEqOp,
                  CmpNeOp,
                  CmpLtOp,
                  CmpGtOp,
                  CmpLeOp,
                  CmpGeOp>(Op))
      return elideArithmeticCasts(Op);
  }

private:
  class OperatorType {
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
    if (IntType.getSize() < DataModel.getIntSize()) {
      T = IntegerType::get(T.getContext(),
                           IntegerKind::Signed,
                           DataModel.getIntSize());
    }

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

    if (mlir::isa<ShiftLeftOp, ShiftRightOp>(Op))
      return { promote(Types.front()), promote(Types.back()) };

    if (mlir::isa<AddOp,
                  SubOp,
                  MulOp,
                  DivOp,
                  RemOp,
                  BitwiseAndOp,
                  BitwiseOrOp,
                  BitwiseXorOp,
                  CmpEqOp,
                  CmpNeOp,
                  CmpLtOp,
                  CmpGtOp,
                  CmpLeOp,
                  CmpGeOp>(Op))
      return getCommonIntegerType(Op->getContext(), Types);

    revng_abort();
  }

  void markAsImplicit(CastOpInterface Cast) {
    Cast->setAttr("clift.implicit", mlir::UnitAttr::get(Cast->getContext()));
  }

  bool isImplicitConversion(CastOpInterface Cast) {
    mlir::Type DstT = unwrapTypedefs(Cast.getResult().getType());
    mlir::Type SrcT = unwrapTypedefs(Cast.getValue().getType());

    if (mlir::isa<IntegralType>(SrcT) and mlir::isa<IntegralType>(DstT))
      return true;

    if (mlir::isa<BitCastOp>(Cast)) {
      auto DstPtrT = mlir::dyn_cast<PointerType>(DstT);
      auto SrcPtrT = mlir::dyn_cast<PointerType>(SrcT);

      if (DstPtrT and SrcPtrT) {
        auto DstPointeeT = collapseTypedefs(DstPtrT.getPointeeType());
        auto SrcPointeeT = collapseTypedefs(SrcPtrT.getPointeeType());

        if (isConst(SrcPointeeT) and not isConst(DstPointeeT))
          return false;

        if (equivalent(SrcPointeeT, DstPointeeT))
          return true;

        if (mlir::isa<VoidType>(DstPointeeT))
          return true;

        return false;
      }
    }

    return false;
  }

  void elideCoercingContextCasts(mlir::OpOperand &Operand) {
    if (auto Cast = Operand.get().getDefiningOp<CastOpInterface>()) {
      if (isImplicitConversion(Cast))
        markAsImplicit(Cast);
    }
  }

  void elideAssignmentCasts(AssignOp Op) {
    elideCoercingContextCasts(Op->getOpOperand(1));
  }

  void elideArgumentCasts(CallOp Op) {
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

      mlir::Type CastOperandType = Cast.getValue().getType();
      if (not unwrapped_isa<IntegralType>(CastOperandType))
        continue;

      ScopedExchange Transaction(T, CastOperandType);
      if (getArithmeticOperatorType(Op, Types) != OperatorType)
        continue;

      Transaction.commit();
      markAsImplicit(Cast);
    }
  }
};

template<typename T>
using PassBase = clift::impl::CliftImplicitCastElisionBase<T>;

struct ImplicitCastElisionPass : PassBase<ImplicitCastElisionPass> {
  void runOnOperation() override {
    FunctionOp Function = getOperation();
    ImplicitCastElider Elider(getDataModel(Function));

    Function->walk([&](mlir::Operation *Op) { Elider.elide(Op); });
  }
};

} // namespace

PassPtr<FunctionOp> clift::createImplicitCastElisionPass() {
  return std::make_unique<ImplicitCastElisionPass>();
}
