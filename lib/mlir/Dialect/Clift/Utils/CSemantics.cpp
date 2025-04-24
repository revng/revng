//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/mlir/Dialect/Clift/Utils/CSemantics.h"
#include "revng/mlir/Dialect/Clift/Utils/ModuleValidator.h"

namespace clift = mlir::clift;
using namespace clift;

namespace {

static clift::PointerType getPointerOperationType(mlir::Operation *Op) {
  auto GetPointerTypeChecked = [&](mlir::Type Type) {
    auto PointerType = getPointerType(Type);
    revng_assert(PointerType);
    return PointerType;
  };

  if (mlir::isa<PtrAddOp, PtrSubOp, AddressofOp>(Op))
    return GetPointerTypeChecked(Op->getResult(0).getType());

  if (mlir::isa<PtrDiffOp, IndirectionOp, SubscriptOp>(Op))
    return GetPointerTypeChecked(Op->getOperand(0).getType());

  if (auto A = mlir::dyn_cast<AccessOp>(Op); A and A.isIndirect())
    return GetPointerTypeChecked(A.getValue().getType());

  if (auto C = mlir::dyn_cast<CastOp>(Op); C and C.getKind() == CastKind::Decay)
    return GetPointerTypeChecked(C.getResult().getType());

  if (auto C = mlir::dyn_cast<CallOp>(Op)) {
    if (auto T = getPointerType(C.getFunction().getType()))
      return T;
  }

  return {};
}

class CVerifier : public ModuleValidator<CVerifier> {
public:
  explicit CVerifier(const TargetCImplementation &Target) : Target(Target) {}

  mlir::LogicalResult visitNestedOp(mlir::Operation *Op) {
    if (auto T = getPointerOperationType(Op)) {
      if (T.getPointerSize() != Target.PointerSize)
        return getCurrentOp()->emitOpError() << "Pointer operation is not "
                                                "representable in the target "
                                                "implementation.";
    }

    if (mlir::isa<ImmediateOp>(Op)) {
      auto T = mlir::cast<ValueType>(Op->getResult(0).getType());

      if (isPotentiallyPromotingType(T))
        return Op->emitOpError() << " is not representable in the target"
                                 << " implementation.";
    }

    if (isPromotingOp(Op)) {
      auto T = mlir::cast<ValueType>(Op->getResult(0).getType());

      if (isPotentiallyPromotingType(T))
        return Op->emitOpError() << " causes integer promotion in the target"
                                    " implementation.";
    }

    if (isBooleanOp(Op)) {
      auto T = mlir::cast<ValueType>(Op->getResult(0).getType());

      if (not isCanonicalBooleanType(T))
        return Op->emitOpError() << " - not yielding the canonical boolean type"
                                 << " - is not representable in the target"
                                 << " implementation.";
    }

    return mlir::success();
  }

private:
  static bool isPromotingOp(mlir::Operation *Op) {
    return mlir::isa<NegOp,
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
                     ShiftRightOp>(Op);
  }

  static bool isBooleanOp(mlir::Operation *Op) {
    return mlir::isa<LogicalNotOp,
                     LogicalAndOp,
                     LogicalOrOp,
                     CmpEqOp,
                     CmpNeOp,
                     CmpLtOp,
                     CmpGtOp,
                     CmpLeOp,
                     CmpGeOp>(Op);
  }

  bool isPotentiallyPromotingType(ValueType Type) {
    if (auto P = mlir::dyn_cast<PrimitiveType>(dealias(Type, true))) {
      if (isIntegerKind(P.getKind())) {
        auto Integer = Target.getIntegerKind(P.getSize());
        return not Integer or *Integer < CIntegerKind::Int;
      }
    }
    return false;
  }

  bool isCanonicalBooleanType(ValueType Type) {
    if (auto P = mlir::dyn_cast<PrimitiveType>(dealias(Type, true))) {
      if (isIntegerKind(P.getKind()))
        return Target.getIntegerKind(P.getSize()) == CIntegerKind::Int;
    }
    return false;
  }

  const TargetCImplementation &Target;
};

} // namespace

mlir::LogicalResult
clift::verifyCSemantics(mlir::ModuleOp Module,
                        const TargetCImplementation &Target) {
  return CVerifier::validate(Module, Target);
}
