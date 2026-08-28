//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/ModuleVisitor.h"
#include "revng/CliftEmitC/CSemantics.h"

using namespace clift;

namespace {

static PointerType getPointerOperationType(ExpressionOpInterface Op) {
  if (mlir::isa<PtrAddOp, PtrSubOp, AddressofOp>(Op))
    return clift::unwrapped_cast<PointerType>(Op.getType());

  if (mlir::isa<PtrDiffOp, IndirectionOp, SubscriptOp>(Op))
    return clift::unwrapped_cast<PointerType>(Op->getOperand(0).getType());

  if (auto A = mlir::dyn_cast<IndirectAccessOp>(Op.getOperation()))
    return clift::unwrapped_cast<PointerType>(A.getValue().getType());

  if (mlir::isa<DecayOp>(Op))
    return clift::unwrapped_cast<PointerType>(Op.getType());

  if (auto C = mlir::dyn_cast<CallOp>(Op.getOperation())) {
    if (auto T = clift::unwrapped_dyn_cast<PointerType>(C.getFunction()
                                                          .getType()))
      return T;
  }

  return {};
}

class CVerifier : public ModuleVisitor<CVerifier> {
  std::optional<CDataModel> DataModel;

public:
  mlir::LogicalResult visitNestedOp(mlir::Operation *Op) {
    auto E = mlir::dyn_cast<ExpressionOpInterface>(Op);
    if (not E)
      return mlir::success();

    if (auto T = getPointerOperationType(E)) {
      if (T.getPointerSize() != DataModel->PointerSize)
        return getCurrentOp()->emitOpError() << "Pointer operation is not "
                                                "representable in the target "
                                                "implementation.";
    }

    if (mlir::isa<ImmediateOp>(E)) {
      if (isPotentiallyPromotingType(E.getType()))
        return Op->emitOpError() << " is not representable in the target"
                                 << " implementation.";
    }

    if (isPromotingOp(E)) {
      if (isPotentiallyPromotingType(E.getType()))
        return Op->emitOpError() << " causes integer promotion in the target"
                                    " implementation.";
    }

    if (isBooleanOp(E)) {
      if (not isCanonicalBooleanType(E.getType()))
        return Op->emitOpError() << " - not yielding the canonical boolean type"
                                 << " - is not representable in the target"
                                 << " implementation.";
    }

    if (hasMismatchedSignedness(E))
      return Op->emitOpError() << " operand signedness does not match operation"
                                  " semantics.";

    return mlir::success();
  }

  mlir::LogicalResult visitModuleOp(mlir::ModuleOp Op) {
    DataModel = getDataModel(Op);
    return mlir::success();
  }

private:
  static bool isPromotingOp(mlir::Operation *Op) {
    return mlir::isa<NegOp,
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
                     SarOp>(Op);
  }

  static bool isBooleanOp(mlir::Operation *Op) {
    return mlir::isa<LogicalNotOp,
                     LogicalAndOp,
                     LogicalOrOp,
                     CmpEqOp,
                     CmpNeOp,
                     SCmpLtOp,
                     UCmpLtOp,
                     SCmpGtOp,
                     UCmpGtOp,
                     SCmpLeOp,
                     UCmpLeOp,
                     SCmpGeOp,
                     UCmpGeOp>(Op);
  }

  static bool hasMismatchedSignedness(mlir::Operation *Op) {
    if (mlir::isa<ShrOp,
                  UDivOp,
                  URemOp,
                  UCmpLtOp,
                  UCmpGtOp,
                  UCmpLeOp,
                  UCmpGeOp>(Op))
      return clift::isSigned(Op->getOperand(0).getType());

    if (mlir::isa<SarOp,
                  SDivOp,
                  SRemOp,
                  SCmpLtOp,
                  SCmpGtOp,
                  SCmpLeOp,
                  SCmpGeOp>(Op))
      return not clift::isSigned(Op->getOperand(0).getType());

    return false;
  }

  bool isPotentiallyPromotingType(mlir::Type Type) {
    if (auto IntType = clift::unwrapped_dyn_cast<IntegerType>(Type))
      return IntType.getSize() < DataModel->getIntSize();
    return false;
  }

  bool isCanonicalBooleanType(mlir::Type Type) {
    if (auto IntType = clift::unwrapped_dyn_cast<IntegerType>(Type))
      return IntType.getSize() == DataModel->getIntSize();
    return false;
  }
};

} // namespace

mlir::LogicalResult verifyCSemantics(mlir::ModuleOp Module) {
  return CVerifier::visit(Module);
}
