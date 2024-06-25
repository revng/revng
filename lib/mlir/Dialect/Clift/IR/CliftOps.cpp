//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ScopeExit.h"

#include "mlir/IR/RegionGraphTraits.h"

#include "revng/Support/GraphAlgorithms.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"

#define GET_OP_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.cpp.inc"

#include "CliftTypeHelpers.h"

using namespace mlir;
using namespace mlir::clift;

void mlir::clift::CliftDialect::registerOperations() {
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

//===-----------------------------------------------------------------========//
// Code for clift::AssignLabelOp.
//===----------------------------------------------------------------------===//

mlir::LogicalResult
mlir::clift::AssignLabelOp::canonicalize(mlir::clift::AssignLabelOp Op,
                                         mlir::PatternRewriter &Rewriter) {
  for (const mlir::OpOperand &Use : Op.getLabel().getUses())
    if (mlir::isa<mlir::clift::GoToOp>(Use.getOwner()))
      return mlir::success();

  Rewriter.eraseOp(Op);
  return mlir::success();
}

//===-----------------------------------------------------------------========//
// Code for clift::MakeLabelOp.
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::clift::MakeLabelOp::verify() {

  auto AssignOwnerLambda = [](mlir::OpOperand &Operand) {
    return mlir::isa<mlir::clift::AssignLabelOp>(Operand.getOwner());
  };
  const bool HasAssign = llvm::any_of(getResult().getUses(), AssignOwnerLambda);

  if (HasAssign)
    return mlir::success();

  auto GoToOwnerLambda = [](mlir::OpOperand &Operand) {
    return mlir::isa<mlir::clift::GoToOp>(Operand.getOwner());
  };
  const bool HasGoTo = llvm::any_of(getResult().getUses(), GoToOwnerLambda);

  if (not HasGoTo)
    return mlir::success();

  emitOpError(getOperationName() + " with a "
              + mlir::clift::GoToOp::getOperationName() + " use must have a "
              + mlir::clift::AssignLabelOp::getOperationName() + " use too.");
  return mlir::failure();
}

//===-----------------------------------------------------------------========//
// Code for clift::LoopOp.
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::clift::LoopOp::verifyRegions() {

  // Verify that the region inside each `clift.loop` is acyclic

  // TODO: this verifier does not cover the root region, because that is not
  //       inside a `clift.loop`. A solution to this may be the definition of a
  //       `IsCombableInterface` in order to perform the verification on the
  //       interface and not on the operation.
  mlir::Region &LoopOpRegion = getBody();
  if (isDAG(&LoopOpRegion)) {
    return success();
  } else {
    return failure();
  }
}

//===-----------------------------------------------------------------========//
// Code for clift::ModuleOp.
//===----------------------------------------------------------------------===//

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
