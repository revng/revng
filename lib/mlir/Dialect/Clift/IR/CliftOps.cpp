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

static YieldOp getExpressionYieldOp(Region &R) {
  if (R.empty())
    return {};

  Block &B = R.front();

  if (B.empty())
    return {};

  return mlir::dyn_cast<clift::YieldOp>(B.back());
}

static ValueType getExpressionType(Region &R) {
  if (auto Yield = getExpressionYieldOp(R)) {
    return Yield.getValue().getType();
  }
  return {};
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

  return R.empty() or static_cast<bool>(getExpressionYieldOp(R));
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

//===-------------------------- GlobalVariableOp --------------------------===//

mlir::LogicalResult GlobalVariableOp::verify() {
  if (Region &R = getInitializer(); not R.empty()) {
    if (getExpressionType(R) != getType())
      return emitOpError() << getOperationName()
                           << " initializer type must match the variable type";
  }

  return mlir::success();
}

//===----------------------------- Statements -----------------------------===//

//===------------------------------ DoWhileOp -----------------------------===//

mlir::LogicalResult DoWhileOp::verify() {
  if (not isScalarType(getExpressionType(getCondition())))
    return emitOpError() << getOperationName()
                         << " condition requires a scalar type.";

  return mlir::success();
}

//===-------------------------------- ForOp -------------------------------===//

mlir::LogicalResult ForOp::verify() {
  Region &Initializer = getInitializer();

  if (not Initializer.empty()) {
    // TODO: Decide what should be accepted in a for-loop init-statement and
    //       Implement verification of it.
    return emitOpError() << getOperationName()
                         << " init statements are not yet supported.";
  }

  if (auto ConditionType = getExpressionType(getCondition())) {
    if (not isScalarType(ConditionType))
      return emitOpError() << getOperationName()
                           << " condition requires a scalar type.";
  }

  return mlir::success();
}

//===-------------------------------- IfOp --------------------------------===//

mlir::LogicalResult IfOp::verify() {
  if (not isScalarType(getExpressionType(getCondition())))
    return emitOpError() << getOperationName()
                         << " condition requires a scalar type.";

  return mlir::success();
}

//===--------------------------- LocalVariableOp --------------------------===//

mlir::LogicalResult LocalVariableOp::verify() {
  if (Region &R = getInitializer(); not R.empty()) {
    if (getExpressionType(R) != getType())
      return emitOpError() << getOperationName()
                           << " initializer type must match the variable type";
  }

  return mlir::success();
}

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

//===------------------------------ ReturnOp ------------------------------===//

mlir::LogicalResult ReturnOp::verify() {
  if (not verifyFunctionReturnType(getExpressionType(getResult())))
    return emitOpError() << getOperationName()
                         << " requires void or non-array object type.";

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

mlir::ParseResult SwitchOp::parse(OpAsmParser &Parser, OperationState &Result) {
  // Condition region:
  Result.addRegion(std::make_unique<Region>());

  // Default case region:
  Result.addRegion(std::make_unique<Region>());

  if (Parser.parseRegion(*Result.regions[0]).failed())
    return Parser.emitError(Parser.getCurrentLocation(),
                            "Expected switch condition region");

  llvm::SmallVector<int64_t, 16> CaseValues;
  while (Parser.parseOptionalKeyword("case").succeeded()) {
    uint64_t CaseValue;
    if (Parser.parseInteger(CaseValue).failed())
      return Parser.emitError(Parser.getCurrentLocation(),
                              "Expected switch case value");

    auto R = std::make_unique<Region>();
    if (Parser.parseRegion(*R).failed())
      return Parser.emitError(Parser.getCurrentLocation(),
                              "Expected switch case region");

    CaseValues.push_back(static_cast<uint64_t>(CaseValue));
    Result.addRegion(std::move(R));
  }

  if (Parser.parseOptionalKeyword("default").succeeded()) {
    if (Parser.parseRegion(*Result.regions[1]).failed())
      return Parser.emitError(Parser.getCurrentLocation(),
                              "Expected switch default region");
  }

  Result.attributes.set("case_values",
                        DenseI64ArrayAttr::get(Parser.getContext(),
                                               CaseValues));

  if (Parser.parseOptionalAttrDict(Result.attributes).failed())
    return mlir::failure();

  return mlir::success();
}

void SwitchOp::print(OpAsmPrinter &Printer) {
  Printer << ' ';
  Printer.printRegion(getConditionRegion());

  for (unsigned I = 0, C = getNumCases(); I < C; ++I) {
    Printer << " case " << getCaseValue(I) << ' ';
    Printer.printRegion(getCaseRegion(I));
  }

  if (hasDefaultCase()) {
    Printer << " default ";
    Printer.printRegion(getDefaultCaseRegion());
  }

  static constexpr llvm::StringRef Elided[] = {
    "case_values",
  };

  Printer.printOptionalAttrDict(getOperation()->getAttrs(), Elided);
}

mlir::LogicalResult SwitchOp::verify() {
  if (not isIntegerType(getExpressionType(getCondition())))
    return emitOpError() << getOperationName()
                         << " condition requires an integer type.";

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

//===------------------------------- WhileOp ------------------------------===//

mlir::LogicalResult WhileOp::verify() {
  if (not isScalarType(getExpressionType(getCondition())))
    return emitOpError() << getOperationName()
                         << " condition requires a scalar type.";

  return mlir::success();
}

//===----------------------------- Expressions ----------------------------===//

//===------------------------------- YieldOp ------------------------------===//

mlir::LogicalResult YieldOp::verify() {
  auto T = getValue().getType();

  if (not isObjectType(T) and not isVoid(T))
    return emitOpError() << getOperationName()
                         << " must yield a non-void object type.";

  return mlir::success();
}
