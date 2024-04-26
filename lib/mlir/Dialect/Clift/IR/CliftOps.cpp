//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/RegionGraphTraits.h"

#include "revng/Support/GraphAlgorithms.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"

#define GET_OP_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.cpp.inc"

void mlir::clift::CliftDialect::registerOperations() {
  addOperations</* Include the auto-generated clift operations */
#define GET_OP_LIST
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.cpp.inc"
                /* End of operations list */>();
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

struct ModuleValidator {
  mlir::LogicalResult visitSingleType(mlir::Operation *ContainingOp,
                                      mlir::Type Type) {
    if (checkTypeIsAcceptable(Type, ContainingOp).failed())
      return mlir::failure();

    auto Casted = Type.dyn_cast<mlir::clift::DefinedType>();
    if (not Casted)
      return mlir::success();
    if (auto Iter = Definitions.find(Casted.id());
        Iter != Definitions.end() and Iter->second != Casted.getElementType()) {
      ContainingOp->emitError("Found two distinct type definitions with the "
                              "same ID");
      return mlir::failure();
    }
    Definitions[Casted.id()] = Casted.getElementType();

    return mlir::success();
  }

  mlir::LogicalResult visitType(mlir::Operation *ContainingOp,
                                mlir::Type Type) {
    if (VisistedTypes.contains(Type))
      return mlir::success();
    VisistedTypes.insert(Type);

    if (visitSingleType(ContainingOp, Type).failed())
      return mlir::failure();

    auto Casted = Type.dyn_cast<mlir::SubElementTypeInterface>();
    if (not Casted) {
      return mlir::success();
    }
    mlir::LogicalResult Result = mlir::success();
    const auto WalkType = [&, this](mlir::Type Inner) {
      if (visitType(ContainingOp, Inner).failed())
        Result = mlir::failure();
    };
    const auto WalkAttr = [&, this](mlir::Attribute Attr) {
      if (visitAttr(ContainingOp, Attr).failed())
        Result = mlir::failure();
    };
    Casted.walkImmediateSubElements(WalkAttr, WalkType);
    return Result;
  }

  mlir::LogicalResult visitAttr(mlir::Operation *ContainingOp,
                                mlir::Attribute Attr) {
    if (VisitedAttrs.contains(Attr))
      return mlir::success();
    VisitedAttrs.insert(Attr);

    auto Casted = Attr.dyn_cast<mlir::SubElementAttrInterface>();
    if (not Casted) {
      return mlir::success();
    }
    mlir::LogicalResult Result = mlir::success();
    const auto WalkType = [&, this](mlir::Type Inner) {
      if (visitType(ContainingOp, Inner).failed())
        Result = mlir::failure();
    };
    const auto WalkAttr = [&, this](mlir::Attribute Attr) {
      if (visitAttr(ContainingOp, Attr).failed())
        Result = mlir::failure();
    };
    Casted.walkImmediateSubElements(WalkAttr, WalkType);
    return Result;
  }

  mlir::LogicalResult checkTypeIsAcceptable(mlir::Type Type,
                                            mlir::Operation *Op) {
    if (not Type.isa<mlir::clift::LabelType>()
        and not Type.isa<mlir::clift::ValueType>()) {
      Op->emitError("Only label types and value types are accepted in a "
                    "clift module value");
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
  llvm::SmallPtrSet<mlir::Type, 2> VisistedTypes;
  llvm::SmallPtrSet<mlir::Attribute, 2> VisitedAttrs;
  llvm::DenseMap<size_t, mlir::clift::TypeDefinitionAttr> Definitions;
};

mlir::LogicalResult mlir::clift::ModuleOp::verify() {

  ModuleValidator Validator;
  if (getBody()
        .walk([&](Operation *Op) {
          if (Validator.visitOp(Op).failed())
            return mlir::WalkResult::interrupt();
          return mlir::WalkResult::advance();
        })
        .wasInterrupted()) {
    return mlir::failure();
  }

  return Validator.visitOp(*this);
}
