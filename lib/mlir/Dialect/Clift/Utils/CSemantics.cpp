//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/mlir/Dialect/Clift/Utils/CSemantics.h"
#include "revng/mlir/Dialect/Clift/Utils/ModuleValidator.h"

namespace clift = mlir::clift;
using namespace clift;

namespace {

class CVerifier : public ModuleValidator<CVerifier> {
public:
  explicit CVerifier(const TargetCImplementation &Target) : Target(Target) {}

  mlir::LogicalResult visitType(mlir::Type Type) {
    if (auto T = mlir::dyn_cast<PointerType>(Type)) {
      if (T.getPointerSize() != Target.PointerSize)
        return getCurrentOp()->emitOpError() << "Pointer type is not "
                                                "representable in the target "
                                                "implementation.";
    }

    return mlir::success();
  }

  mlir::LogicalResult visitNestedOp(mlir::Operation *Op) {
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
                     EqualOp,
                     NotEqualOp,
                     LessThanOp,
                     GreaterThanOp,
                     LessThanOrEqualOp,
                     GreaterThanOrEqualOp>(Op);
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
clift::verifyCSemantics(clift::ModuleOp Module,
                        const TargetCImplementation &Target) {
  return CVerifier::validate(Module, Target);
}
