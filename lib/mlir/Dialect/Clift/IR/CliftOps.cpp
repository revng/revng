//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"

#include "mlir/IR/RegionGraphTraits.h"

#include "revng/Support/GraphAlgorithms.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"

#define GET_OP_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.cpp.inc"

#include "CliftTypeHelpers.h"

using namespace mlir;
using namespace mlir::clift;

void CliftDialect::registerOperations() {
  addOperations</* Include the auto-generated clift operations */
#define GET_OP_LIST
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.cpp.inc"
                /* End of operations list */>();
}

//===---------------------------- Region types ----------------------------===//

template<typename OpInterface>
static bool verifyRegionContent(Region &R, const bool Required) {
  if (R.empty())
    return not Required;

  if (not R.hasOneBlock())
    return false;

  for (Operation &Op : R.front()) {
    if (not mlir::isa<OpInterface>(&Op))
      return false;
  }

  return true;
}

bool clift::impl::verifyStatementRegion(Region &R) {
  return verifyRegionContent<StatementOpInterface>(R, false);
}

bool clift::impl::verifyExpressionRegion(Region &R, const bool Required) {
  if (not verifyRegionContent<ExpressionOpInterface>(R, Required))
    return false;

  if (not R.empty()) {
    Block &B = R.front();

    if (B.empty())
      return false;
  }

  return true;
}

//===------------------------------ ModuleOp ------------------------------===//

void clift::ModuleOp::build(OpBuilder &Builder, OperationState &State) {
  State.addRegion()->emplaceBlock();
}

namespace {

struct ModuleValidator {
  template<typename SubElementInterface>
  mlir::LogicalResult visitSubElements(mlir::Operation *ContainingOp,
                                       SubElementInterface Interface) {
    mlir::LogicalResult R = mlir::success();

    const auto WalkType = [&](mlir::Type InnerType) {
      if (visitType(ContainingOp, InnerType).failed())
        R = mlir::failure();
    };
    const auto WalkAttr = [&](mlir::Attribute InnerAttr) {
      if (visitAttr(ContainingOp, InnerAttr).failed())
        R = mlir::failure();
    };
    Interface.walkImmediateSubElements(WalkAttr, WalkType);

    return R;
  };

  // Visit a field type of a class type attribute.
  // RootAttr is the root class type attribute and is used to detect recursion.
  mlir::LogicalResult visitFieldType(mlir::Operation *ContainingOp,
                                     clift::ValueType FieldType,
                                     TypeDefinitionAttr RootAttr) {
    FieldType = dealias(FieldType);

    if (auto T = mlir::dyn_cast<DefinedType>(FieldType)) {
      if (T.getElementType() == RootAttr)
        return ContainingOp->emitError() << "Clift ModuleOp contains a "
                                            "recursive class type.";

      return maybeVisitClassTypeAttr(ContainingOp,
                                     T.getElementType(),
                                     RootAttr);
    }

    return mlir::success();
  }

  template<typename ClassTypeAttr>
  mlir::LogicalResult visitClassTypeAttr(mlir::Operation *ContainingOp,
                                         ClassTypeAttr Attr,
                                         TypeDefinitionAttr RootAttr) {
    for (FieldAttr Field : Attr.getFields()) {
      if (visitFieldType(ContainingOp, Field.getType(), RootAttr).failed())
        return mlir::failure();
    }
    return mlir::success();
  }

  // Call visitClassTypeAttr if Attr is a class type attribute.
  // RootAttr is the root class type attribute and is used to detect recursion.
  mlir::LogicalResult maybeVisitClassTypeAttr(mlir::Operation *ContainingOp,
                                              TypeDefinitionAttr Attr,
                                              TypeDefinitionAttr RootAttr) {
    if (auto T = mlir::dyn_cast<StructTypeAttr>(Attr))
      return visitClassTypeAttr(ContainingOp, T, RootAttr);
    if (auto T = mlir::dyn_cast<UnionTypeAttr>(Attr))
      return visitClassTypeAttr(ContainingOp, T, RootAttr);
    return mlir::success();
  }

  mlir::LogicalResult visitTypeAttr(mlir::Operation *ContainingOp,
                                    TypeDefinitionAttr Attr) {
    auto const [Iterator, Inserted] = Definitions.try_emplace(Attr.id(), Attr);

    if (not Inserted and Iterator->second != Attr)
      return ContainingOp->emitError() << "Found two distinct type definitions "
                                          "with the same ID";

    if (maybeVisitClassTypeAttr(ContainingOp, Attr, Attr).failed())
      return mlir::failure();

    return mlir::success();
  }

  mlir::LogicalResult visitValueType(mlir::Operation *ContainingOp,
                                     clift::ValueType Type) {
    Type = dealias(Type);

    if (not isCompleteType(Type))
      return ContainingOp->emitError() << "Clift ModuleOp contains an "
                                          "incomplete type";

    if (auto T = mlir::dyn_cast<DefinedType>(Type)) {
      if (visitTypeAttr(ContainingOp, T.getElementType()).failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult visitSingleType(mlir::Operation *ContainingOp,
                                      mlir::Type Type) {
    if (Type.getDialect().getTypeID() != mlir::TypeID::get<CliftDialect>())
      return ContainingOp->emitError() << "Clift ModuleOp a contains non-Clift "
                                          "type";

    if (auto T = mlir::dyn_cast<ValueType>(Type)) {
      if (visitValueType(ContainingOp, Type).failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult visitType(mlir::Operation *ContainingOp,
                                mlir::Type Type) {
    if (not VisitedTypes.insert(Type).second)
      return mlir::success();

    if (visitSingleType(ContainingOp, Type).failed())
      return mlir::failure();

    if (auto T = mlir::dyn_cast<mlir::SubElementTypeInterface>(Type)) {
      if (visitSubElements(ContainingOp, T).failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult visitAttr(mlir::Operation *ContainingOp,
                                mlir::Attribute Attr) {
    if (not VisitedAttrs.insert(Attr).second)
      return mlir::success();

    if (auto T = mlir::dyn_cast<mlir::SubElementAttrInterface>(Attr)) {
      if (visitSubElements(ContainingOp, T).failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult visitOp(mlir::Operation *Op) {
    for (const auto &Result : Op->getOpResults()) {
      if (visitType(Op, Result.getType()).failed())
        return mlir::failure();
    }

    for (mlir::Region &Region : Op->getRegions()) {
      for (auto &Block : Region.getBlocks()) {
        for (auto &Argument : Block.getArguments()) {
          if (visitType(Op, Argument.getType()).failed())
            return mlir::failure();
        }
      }
    }

    for (auto Attr : Op->getAttrs()) {
      if (visitAttr(Op, Attr.getValue()).failed())
        return mlir::failure();
    }

    return mlir::success();
  }

private:
  llvm::SmallPtrSet<mlir::Type, 32> VisitedTypes;
  llvm::SmallPtrSet<mlir::Attribute, 32> VisitedAttrs;
  llvm::DenseMap<uint64_t, TypeDefinitionAttr> Definitions;
};

} // namespace

mlir::LogicalResult clift::ModuleOp::verify() {
  ModuleValidator Validator;

  const auto Visitor = [&](Operation *Op) -> mlir::WalkResult {
    return Validator.visitOp(Op);
  };

  if (walk(Visitor).wasInterrupted())
    return mlir::failure();

  return Validator.visitOp(*this);
}

//===----------------------------- Statements -----------------------------===//

//===----------------------------- MakeLabelOp ----------------------------===//

static std::pair<size_t, size_t> getNumLabelUsers(MakeLabelOp Op) {
  size_t Assignments = 0;
  size_t GoTos = 0;
  for (mlir::OpOperand &Operand : Op.getResult().getUses()) {
    if (mlir::isa<AssignLabelOp>(Operand.getOwner()))
      ++Assignments;
    else if (mlir::isa<GoToOp>(Operand.getOwner()))
      ++GoTos;
  }
  return { Assignments, GoTos };
}

mlir::LogicalResult MakeLabelOp::canonicalize(MakeLabelOp Op,
                                              PatternRewriter &Rewriter) {
  const auto [Assignments, GoTos] = getNumLabelUsers(Op);

  if (GoTos != 0)
    return mlir::success();

  for (mlir::OpOperand &Operand : Op.getResult().getUses()) {
    if (auto AssignOp = mlir::dyn_cast<AssignLabelOp>(Operand.getOwner()))
      Rewriter.eraseOp(AssignOp);
  }

  Rewriter.eraseOp(Op);
  return mlir::success();
}

mlir::LogicalResult MakeLabelOp::verify() {
  const auto [Assignments, GoTos] = getNumLabelUsers(*this);

  if (Assignments > 1)
    return emitOpError() << getOperationName()
                         << " may only have one assignment.";

  if (GoTos != 0 and Assignments == 0)
    return emitOpError() << getOperationName() << " with a use by "
                         << GoToOp::getOperationName()
                         << " must have an assignment.";

  return mlir::success();
}

//===------------------------------ SwitchOp ------------------------------===//

void SwitchOp::build(OpBuilder &OdsBuilder,
                     OperationState &OdsState,
                     const llvm::ArrayRef<uint64_t> CaseValues) {
  llvm::SmallVector<int64_t> SignedCaseValues;
  SignedCaseValues.resize_for_overwrite(CaseValues.size());
  std::copy(CaseValues.begin(), CaseValues.end(), SignedCaseValues.begin());
  build(OdsBuilder, OdsState, SignedCaseValues, CaseValues.size());
}

mlir::LogicalResult SwitchOp::verify() {
  // One region for the condition, one for the default case and N for others.
  if (getNumRegions() != 2 + getCaseValues().size())
    return emitOpError() << getOperationName()
                         << " must have a case value for each case region.";

  llvm::SmallSet<uint64_t, 16> CaseValueSet;
  for (uint64_t const CaseValue : getCaseValues()) {
    if (not CaseValueSet.insert(CaseValue).second)
      return emitOpError() << getOperationName()
                           << " case values must be unique.";
  }

  return mlir::success();
}
