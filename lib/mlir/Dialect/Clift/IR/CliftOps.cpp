//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"

#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/RegionGraphTraits.h"

#include "revng/Support/GraphAlgorithms.h"
#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng/mlir/Dialect/Clift/Utils/ModuleValidator.h"

#define GET_OP_CLASSES
#include "revng/mlir/Dialect/Clift/IR/CliftOps.cpp.inc"

using namespace mlir;
using namespace mlir::clift;

void CliftDialect::registerOperations() {
  addOperations</* Include the auto-generated clift operations */
#define GET_OP_LIST
#include "revng/mlir/Dialect/Clift/IR/CliftOps.cpp.inc"
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

static FunctionTypeAttr getFunctionTypeAttr(mlir::Type Type) {
  if (auto T = mlir::dyn_cast<DefinedType>(dealias(Type)))
    return mlir::dyn_cast<FunctionTypeAttr>(T.getElementType());
  return {};
}

//===-------------------------- Type constraints --------------------------===//

bool clift::impl::verifyPrimitiveTypeOf(ValueType Type, PrimitiveKind Kind) {
  if (auto T = mlir::dyn_cast<PrimitiveType>(Type))
    return T.getKind() == Kind;

  return false;
}

mlir::Type clift::impl::removeCliftConst(mlir::Type Type) {
  if (auto ValueT = mlir::dyn_cast<ValueType>(Type))
    Type = ValueT.removeConst();

  return Type;
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

//===-------------------------- Operation parsing -------------------------===//

template<typename TypeOrPointer>
static Type deduceResultType(llvm::ArrayRef<TypeOrPointer> Arguments) {
  const auto getType = [](TypeOrPointer Argument) -> ValueType {
    if constexpr (std::is_same_v<TypeOrPointer, Type>) {
      return mlir::cast<ValueType>(Argument);
    } else {
      return mlir::cast<ValueType>(*Argument);
    }
  };

  auto CommonType = getType(Arguments.front()).removeConst();
  for (TypeOrPointer Argument : Arguments.slice(0)) {
    if (getType(Argument).removeConst() != CommonType)
      return {};
  }

  return CommonType;
}

/// Parses one or more operand types optionally followed by a result type.
///
/// The argument types can be specified in two forms:
///   * a single type, or
///   * one or more types separated by commas and delimited by parentheses.
///
/// If no parentheses are used, the single specified argument type is used for
/// all expected argument types. Otherwise the number of specified argument
/// types must match the number of expected argument types.
///
/// The trailing result type is only accepted when @p Result is not null. When
/// the result type is not specified, a default type is deduced by taking each
/// argument types T and removing const to produce the unqualified type U. If
/// all U are equal, then U is deduced. Otherwise the deduction is ambiguous
/// and the parse fails.
///
/// Examples:
///   - !a
///   - !a -> !c
///   - (!a, !b)
///   - (!a, !b) -> !c
ParseResult
mlir::clift::impl::parseCliftOpTypes(OpAsmParser &Parser,
                                     Type *Result,
                                     llvm::ArrayRef<Type *> Arguments) {
  Type &FirstArgument = *Arguments.front();
  if (Parser.parseOptionalLParen().succeeded()) {
    if (Parser.parseType(FirstArgument).failed())
      return mlir::failure();

    for (Type *Argument : Arguments.slice(1)) {
      if (Parser.parseComma().failed())
        return mlir::failure();
      if (Parser.parseType(*Argument).failed())
        return mlir::failure();
    }

    if (Parser.parseRParen().failed())
      return mlir::failure();
  } else {
    if (Parser.parseType(FirstArgument).failed())
      return mlir::failure();

    for (Type *Argument : Arguments.slice(1))
      *Argument = FirstArgument;
  }

  if (Result != nullptr) {
    if (Parser.parseOptionalArrow().succeeded()) {
      if (Parser.parseType(*Result).failed())
        return mlir::failure();
    } else if (ValueType Deduced = deduceResultType(Arguments)) {
      *Result = Deduced;
    } else {
      return Parser.emitError(Parser.getCurrentLocation(),
                              "expected arrow followed by result type");
    }
  }

  return mlir::success();
}

/// @see parseCliftOpTypes for a general description of the syntax.
///
/// If all argument types are equal, a single argument is printed. Otherwise
/// multiple arguments delimited by parentheses are printed.
///
/// If @p Result is not null and it cannot be deduced from the argument types,
/// a trailing type is printed.
void mlir::clift::impl::printCliftOpTypes(OpAsmPrinter &Printer,
                                          Type Result,
                                          llvm::ArrayRef<Type> Arguments) {
  bool ArgumentsEqual = llvm::all_equal(Arguments);
  Type FirstArgument = Arguments.front();

  if (ArgumentsEqual) {
    Printer << FirstArgument;
  } else {
    Printer << "(";
    Printer << FirstArgument;
    for (Type Argument : Arguments.slice(1)) {
      Printer << ", ";
      Printer << Argument;
    }
    Printer << ")";
  }

  if (Result) {
    bool IsDeducible = false;
    if (ArgumentsEqual) {
      if (Result == FirstArgument) {
        IsDeducible = true;
      } else {
        auto FirstArgumentT = mlir::cast<ValueType>(FirstArgument);
        IsDeducible = Result == FirstArgumentT.removeConst();
      }
    } else {
      IsDeducible = Result == deduceResultType(Arguments);
    }

    if (not IsDeducible) {
      Printer << " -> ";
      Printer << Result;
    }
  }
}

//===------------------------------ ModuleOp ------------------------------===//

void clift::ModuleOp::build(OpBuilder &Builder, OperationState &State) {
  State.addRegion()->emplaceBlock();
}

namespace {

static bool isModuleLevelOperation(Operation *Op) {
  if (mlir::isa<clift::FunctionOp>(Op))
    return true;

  if (mlir::isa<clift::GlobalVariableOp>(Op))
    return true;

  return false;
}

class ModuleVerifier : public ModuleValidator<ModuleVerifier> {
  enum class LoopOrSwitch : uint8_t {
    Loop,
    Switch,
  };

public:
  // Visit a field type of a class type attribute.
  // RootAttr is the root class type attribute and is used to detect recursion.
  mlir::LogicalResult visitFieldType(clift::ValueType FieldType,
                                     TypeDefinitionAttr RootAttr) {
    FieldType = dealias(FieldType);

    if (auto T = mlir::dyn_cast<DefinedType>(FieldType)) {
      if (T.getElementType() == RootAttr)
        return getCurrentOp()->emitError() << "Clift ModuleOp contains a "
                                              "recursive class type.";

      return maybeVisitClassTypeAttr(T.getElementType(), RootAttr);
    }

    return mlir::success();
  }

  template<typename ClassTypeAttr>
  mlir::LogicalResult
  visitClassTypeAttr(ClassTypeAttr Attr, TypeDefinitionAttr RootAttr) {
    for (FieldAttr Field : Attr.getFields()) {
      if (visitFieldType(Field.getType(), RootAttr).failed())
        return mlir::failure();
    }
    return mlir::success();
  }

  // Call visitClassTypeAttr if Attr is a class type attribute.
  // RootAttr is the root class type attribute and is used to detect recursion.
  mlir::LogicalResult maybeVisitClassTypeAttr(TypeDefinitionAttr Attr,
                                              TypeDefinitionAttr RootAttr) {
    if (auto T = mlir::dyn_cast<StructTypeAttr>(Attr))
      return visitClassTypeAttr(T, RootAttr);
    if (auto T = mlir::dyn_cast<UnionTypeAttr>(Attr))
      return visitClassTypeAttr(T, RootAttr);
    return mlir::success();
  }

  mlir::LogicalResult visitTypeAttr(TypeDefinitionAttr Attr) {
    auto const [Iterator,
                Inserted] = Definitions.try_emplace(Attr.getUniqueHandle(),
                                                    Attr);

    if (not Inserted and Iterator->second != Attr)
      return getCurrentOp()->emitError() << "Found two distinct type "
                                            "definitions with the same unique "
                                            "handle: '"
                                         << Attr.getUniqueHandle() << '\'';

    if (maybeVisitClassTypeAttr(Attr, Attr).failed())
      return mlir::failure();

    return mlir::success();
  }

  mlir::LogicalResult visitValueType(clift::ValueType Type) {
    Type = dealias(Type);

    if (not isCompleteType(Type))
      return getCurrentOp()->emitError() << "Clift ModuleOp contains an "
                                            "incomplete type";

    if (auto T = mlir::dyn_cast<DefinedType>(Type)) {
      if (visitTypeAttr(T.getElementType()).failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult visitType(mlir::Type Type) {
    if (Type.getDialect().getTypeID() != mlir::TypeID::get<CliftDialect>())
      return getCurrentOp()->emitError() << "Clift ModuleOp a contains "
                                            "non-Clift type";

    if (auto T = mlir::dyn_cast<ValueType>(Type)) {
      if (visitValueType(Type).failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult visitNestedOp(mlir::Operation *Op) {
    if (isModuleLevelOperation(Op))
      return Op->emitOpError() << Op->getName()
                               << " must be directly nested within a"
                                  " ModuleOp.";

    if (auto Return = mlir::dyn_cast<ReturnOp>(Op)) {
      ValueType ReturnType = {};

      if (Region &R = Return.getResult(); not R.empty())
        ReturnType = getExpressionType(R);

      if (ReturnType and isVoid(FunctionReturnType))
        return Op->emitOpError() << Op->getName()
                                 << " cannot return expression in function"
                                    " returning void.";

      if (ReturnType != FunctionReturnType)
        return Op->emitOpError() << Op->getName()
                                 << " type does not match the function return"
                                    " type";
    } else if (mlir::isa<SwitchBreakOp>(Op)) {
      if (not hasLoopOrSwitchParent(Op,
                                    LoopOrSwitch::Switch,
                                    /*DirectlyNested=*/true))
        return Op->emitOpError()
               << Op->getName() << " must be nested within a switch operation.";
    } else if (mlir::isa<LoopBreakOp>(Op)) {
      if (not hasLoopOrSwitchParent(Op,
                                    LoopOrSwitch::Loop,
                                    /*DirectlyNested=*/true))
        return Op->emitOpError()
               << Op->getName() << " must be nested within a loop operation.";
    } else if (mlir::isa<LoopContinueOp>(Op)) {
      if (not hasLoopOrSwitchParent(Op,
                                    LoopOrSwitch::Loop,
                                    /*DirectlyNested=*/false))
        return Op->emitOpError()
               << Op->getName() << " must be nested within a loop operation.";
    } else if (auto Sym = mlir::dyn_cast<MakeLabelOp>(Op)) {
      if (not LabelNames.insert(Sym.getName()).second)
        return Op->emitOpError()
               << Op->getName() << " conflicts with another label.";
    } else if (auto Sym = mlir::dyn_cast<LocalVariableOp>(Op)) {
      if (not LocalNames.insert(Sym.getSymName()).second)
        return Op->emitOpError()
               << Op->getName() << " conflicts with another local variable.";
    }

    return mlir::success();
  }

  mlir::LogicalResult visitModuleLevelOp(mlir::Operation *Op) {
    if (not isModuleLevelOperation(Op))
      return Op->emitOpError() << Op->getName()
                               << " cannot be directly nested within a"
                                  " ModuleOp.";

    if (auto F = mlir::dyn_cast<FunctionOp>(Op)) {
      auto TypeAttr = getFunctionTypeAttr(F.getFunctionType());
      FunctionReturnType = mlir::cast<ValueType>(TypeAttr.getReturnType());

      LocalNames.clear();
      LabelNames.clear();
    }

    return mlir::success();
  }

private:
  clift::ValueType FunctionReturnType;
  llvm::DenseMap<llvm::StringRef, TypeDefinitionAttr> Definitions;

  llvm::DenseSet<llvm::StringRef> LocalNames;
  llvm::DenseSet<llvm::StringRef> LabelNames;

  static std::optional<LoopOrSwitch> isLoopOrSwitch(Operation *Op) {
    if (mlir::isa<ForOp, DoWhileOp, WhileOp>(Op))
      return LoopOrSwitch::Loop;

    if (mlir::isa<SwitchOp>(Op))
      return LoopOrSwitch::Switch;

    return std::nullopt;
  }

  // Finds a loop or switch operation ancestor of the specified op. If
  // DirectlyNested is true, stops at the first such parent found, regardless of
  // its kind. Does not consider other statements, such as if-statements at all.
  bool
  hasLoopOrSwitchParent(Operation *Op, LoopOrSwitch Kind, bool DirectlyNested) {
    while (Op != getCurrentModuleLevelOp()) {
      Op = Op->getParentOp();

      if (auto OpKind = isLoopOrSwitch(Op)) {
        if (*OpKind == Kind)
          return true;

        if (DirectlyNested)
          return false;
      }
    }
    return false;
  }
};

} // namespace

mlir::LogicalResult clift::ModuleOp::verify() {
  if (not getRegion().hasOneBlock())
    return emitOpError() << getOperationName()
                         << " must contain exactly one block.";

  return ModuleVerifier::validate(*this);
}

//===----------------------------- FunctionOp -----------------------------===//

mlir::ParseResult FunctionOp::parse(OpAsmParser &Parser,
                                    OperationState &Result) {
  StringAttr SymbolNameAttr;
  if (Parser
        .parseSymbolName(SymbolNameAttr,
                         SymbolTable::getSymbolAttrName(),
                         Result.attributes)
        .failed())
    return mlir::failure();

  if (Parser.parseLess().failed())
    return mlir::failure();

  auto FunctionTypeLoc = Parser.getCurrentLocation();
  clift::ValueType FunctionType;
  if (Parser.parseType(FunctionType).failed())
    return mlir::failure();

  auto FunctionTypeAttr = ::getFunctionTypeAttr(FunctionType);
  if (not FunctionTypeAttr)
    return Parser.emitError(FunctionTypeLoc) << "expected Clift function or "
                                                "pointer-to-function type.";

  if (Parser.parseGreater().failed())
    return mlir::failure();

  llvm::SmallVector<OpAsmParser::Argument> Arguments;
  llvm::SmallVector<mlir::Type> ResultTypes;
  llvm::SmallVector<DictionaryAttr> ResultAttrs;
  bool IsVariadic = false;

  auto RoughResultTypeLocation = Parser.getCurrentLocation();
  if (function_interface_impl::parseFunctionSignature(Parser,
                                                      /*allowVariadic=*/false,
                                                      Arguments,
                                                      IsVariadic,
                                                      ResultTypes,
                                                      ResultAttrs)
        .failed())
    return mlir::failure();

  if (ResultTypes.size() > 1)
    return Parser.emitError(RoughResultTypeLocation) << "expected no more than "
                                                        "one result";

  auto False = BoolAttr::get(Parser.getContext(), false);

  if (ResultTypes.empty()) {
    ResultTypes.push_back(PrimitiveType::get(Parser.getContext(),
                                             PrimitiveKind::VoidKind,
                                             0,
                                             False));
    ResultAttrs.push_back(DictionaryAttr::get(Parser.getContext()));
  }

  llvm::SmallVector<mlir::Type> ArgumentTypes;
  for (auto &Argument : Arguments)
    ArgumentTypes.push_back(Argument.type);

  Result.addAttribute(getFunctionTypeAttrName(Result.name),
                      TypeAttr::get(FunctionType));

  if (Parser.parseOptionalAttrDictWithKeyword(Result.attributes).failed())
    return mlir::failure();

  function_interface_impl::addArgAndResultAttrs(Parser.getBuilder(),
                                                Result,
                                                Arguments,
                                                ResultAttrs,
                                                getArgAttrsAttrName(Result
                                                                      .name),
                                                getResAttrsAttrName(Result
                                                                      .name));

  auto *Body = Result.addRegion();
  auto RegionParseResult = Parser.parseOptionalRegion(*Body, Arguments);
  if (RegionParseResult.has_value() && mlir::failed(*RegionParseResult))
    return mlir::failure();

  return mlir::success();
}

void FunctionOp::print(OpAsmPrinter &Printer) {
  Printer << ' ';
  Printer.printSymbolName(getSymName());
  Printer << '<';
  Printer.printType(getFunctionType());
  Printer << '>';

  auto FunctionTypeAttr = ::getFunctionTypeAttr(getFunctionType());

  function_interface_impl::printFunctionSignature(Printer,
                                                  *this,
                                                  FunctionTypeAttr
                                                    .getArgumentTypes(),
                                                  /*isVariadic=*/false,
                                                  FunctionTypeAttr
                                                    .getResultTypes());

  function_interface_impl::printFunctionAttributes(Printer,
                                                   *this,
                                                   { getFunctionTypeAttrName(),
                                                     getArgAttrsAttrName(),
                                                     getResAttrsAttrName() });

  if (Region &Body = getBody(); !Body.empty()) {
    Printer << ' ';
    Printer.printRegion(Body,
                        /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
}

ArrayRef<Type> FunctionOp::getArgumentTypes() {
  return ::getFunctionTypeAttr(getFunctionType()).getArgumentTypes();
}

ArrayRef<Type> FunctionOp::getResultTypes() {
  return ::getFunctionTypeAttr(getFunctionType()).getResultTypes();
}

Type FunctionOp::cloneTypeWith(TypeRange inputs, TypeRange results) {
  revng_abort("Operation not supported");
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

//===---------------------------- AssignLabelOp ---------------------------===//

MakeLabelOp AssignLabelOp::getLabelOp() {
  return getLabel().getDefiningOp<MakeLabelOp>();
}

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

//===------------------------------- GotoOp -------------------------------===//

MakeLabelOp GoToOp::getLabelOp() {
  return getLabel().getDefiningOp<MakeLabelOp>();
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
  if (getSymName().empty())
    return emitOpError() << getOperationName()
                         << " must have a non-empty name.";

  if (Region &R = getInitializer(); not R.empty()) {
    if (getExpressionType(R) != getType().removeConst())
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
  if (getName().empty())
    return emitOpError() << getOperationName()
                         << " must have a non-empty name.";

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
  if (not isReturnableType(getExpressionType(getResult())))
    return emitOpError() << getOperationName()
                         << " requires void or non-array object type.";

  return mlir::success();
}

//===------------------------------ SwitchOp ------------------------------===//

ValueType SwitchOp::getConditionType() {
  return getExpressionType(getConditionRegion());
}

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

//===----------------------- UnaryIntegerMutationOp -----------------------===//

mlir::LogicalResult clift::impl::verifyUnaryIntegerMutationOp(Operation *Op) {
  if (not mlir::clift::isLvalueExpression(Op->getOperand(0)))
    return Op->emitOpError()
           << Op->getName() << " operand must be an lvalue-expression.";

  return mlir::success();
}

//===------------------------------- CastOp -------------------------------===//

mlir::LogicalResult CastOp::verify() {
  auto ResT = mlir::cast<ValueType>(getResult().getType());

  if (ResT.isConst())
    return emitOpError() << getOperationName()
                         << " result must have unqualified type.";

  auto ArgT = mlir::cast<ValueType>(getValue().getType());

  switch (auto Kind = getKind()) {
  case CastKind::Extend:
  case CastKind::Truncate: {
    auto ResUnderlyingT = getUnderlyingIntegerType(ResT);
    if (not ResUnderlyingT)
      return emitOpError() << " result must have integer type.";

    auto ArgUnderlyingT = getUnderlyingIntegerType(ArgT);
    if (not ArgUnderlyingT)
      return emitOpError() << " argument must have integer type.";

    if (ResUnderlyingT.getKind() != ArgUnderlyingT.getKind())
      return emitOpError() << " result and argument types must be equal in"
                              " kind.";

    if (Kind == CastKind::Extend) {
      if (ResUnderlyingT.getSize() <= ArgUnderlyingT.getSize())
        return emitOpError() << " result type must be wider than the argument"
                                " type.";
    } else {
      if (ResUnderlyingT.getSize() >= ArgUnderlyingT.getSize())
        return emitOpError() << " result type must be narrower than the"
                                " argument type.";
    }
  } break;
  case CastKind::Reinterpret: {
    if (not isObjectType(ResT) or isArrayType(ResT))
      return emitOpError() << " result must have non-array object type.";

    if (not isObjectType(ArgT) or isArrayType(ArgT))
      return emitOpError() << " argument must have non-array object type.";

    if (ResT.getByteSize() != ArgT.getByteSize())
      return emitOpError() << " result and argument types must be equal in"
                              " size.";
  } break;
  case CastKind::Decay: {
    auto PtrT = mlir::dyn_cast<PointerType>(ResT);
    if (not PtrT)
      return emitOpError() << getOperationName()
                           << " result must have pointer type.";

    if (auto ArrayT = mlir::dyn_cast<ArrayType>(ArgT)) {
      if (PtrT.getPointeeType() != ArrayT.getElementType())
        return emitOpError() << getOperationName()
                             << " the pointee type of the result type must be"
                                " equal to the element type of the argument"
                                " type.";
    } else if (auto DefinedT = mlir::dyn_cast<DefinedType>(ArgT)) {
      auto const
        FunctionT = mlir::dyn_cast<FunctionTypeAttr>(DefinedT.getElementType());

      if (not FunctionT)
        return emitOpError() << getOperationName()
                             << " argument must have array or function type.";

      if (PtrT.getPointeeType() != DefinedT)
        return emitOpError() << getOperationName()
                             << " the pointee type of the result type must be"
                                " equal to the argument type.";
    } else {
      return emitOpError() << getOperationName()
                           << " argument must have array or function type.";
    }
  } break;

  default:
    revng_abort("Invalid CastKind value");
  }

  return mlir::success();
}

//===----------------------------- AddressofOp ----------------------------===//

mlir::LogicalResult AddressofOp::verify() {
  if (not clift::isLvalueExpression(getObject()))
    return emitOpError() << getOperationName()
                         << " operand must be an lvalue-expression.";

  return mlir::success();
}

//===---------------------------- IndirectionOp ---------------------------===//

mlir::LogicalResult IndirectionOp::verify() {
  if (isVoid(getResult().getType()))
    return emitOpError() << getOperationName()
                         << " cannot dereference a pointer to void.";

  return mlir::success();
}

//===------------------------------ AssignOp ------------------------------===//

mlir::LogicalResult AssignOp::verify() {
  if (not clift::isLvalueExpression(getLhs()))
    return emitOpError() << getOperationName()
                         << " left operand must be an lvalue-expression.";

  return mlir::success();
}

//===------------------------------ AccessOp ------------------------------===//

bool AccessOp::isLvalueExpression() {
  return isIndirect() or clift::isLvalueExpression(getValue());
}

DefinedType AccessOp::getClassType() {
  auto ObjectT = dealias(getValue().getType(), /*IgnoreQualifiers=*/true);

  if (isIndirect()) {
    ObjectT = mlir::cast<PointerType>(ObjectT).getPointeeType();
    ObjectT = dealias(ObjectT, /*IgnoreQualifiers=*/true);
  }

  return mlir::cast<DefinedType>(ObjectT);
}

TypeDefinitionAttr AccessOp::getClassTypeAttr() {
  return getClassType().getElementType();
}

FieldAttr AccessOp::getFieldAttr() {
  auto C = mlir::cast<ClassTypeAttr>(getClassTypeAttr());
  return C.getFields()[getMemberIndex()];
}

mlir::LogicalResult AccessOp::verify() {
  auto ObjectT = dealias(getValue().getType());

  if (auto PointerT = mlir::dyn_cast<PointerType>(ObjectT)) {
    if (not isIndirect())
      return emitOpError() << getOperationName()
                           << " operand must have pointer type.";

    ObjectT = dealias(PointerT.getPointeeType(), /*IgnoreQualifiers=*/true);
  }

  auto DefinedT = mlir::dyn_cast<DefinedType>(ObjectT);
  if (not DefinedT)
    return emitOpError() << getOperationName()
                         << " operand must have (pointer to) struct or union"
                         << " type.";

  auto Class = mlir::dyn_cast<ClassTypeAttr>(DefinedT.getElementType());
  if (not Class)
    return emitOpError() << getOperationName()
                         << " operand must have (pointer to) struct or union"
                         << " type.";

  auto Fields = Class.getFields();

  const uint64_t Index = getMemberIndex();
  if (Index >= Fields.size())
    return emitOpError() << getOperationName()
                         << " struct or union member index out of range.";

  auto FieldT = Fields[Index].getType();
  if (FieldT != getResult().getType())
    return emitOpError() << getOperationName()
                         << " result type must match the selected member type.";

  return mlir::success();
}

//===----------------------------- SubscriptOp ----------------------------===//

mlir::LogicalResult SubscriptOp::verify() {
  auto PointerT = mlir::dyn_cast<PointerType>(getPointer().getType());
  if (not PointerT)
    return emitOpError() << getOperationName()
                         << " operand must have pointer type.";

  auto PointeeT = PointerT.getPointeeType();
  if (not isObjectType(PointeeT))
    return emitOpError() << getOperationName()
                         << " cannot dereference pointer to non-object type.";

  if (getResult().getType() != PointeeT)
    return emitOpError() << getOperationName()
                         << " result type must match the pointer type.";

  return mlir::success();
}

//===-------------------------------- UseOp -------------------------------===//

mlir::LogicalResult
UseOp::verifySymbolUses(SymbolTableCollection &SymbolTable) {
  auto Module = getOperation()->getParentOfType<clift::ModuleOp>();
  Operation *Op = SymbolTable.lookupSymbolIn(Module, getSymbolNameAttr());

  if (auto V = mlir::dyn_cast_or_null<GlobalVariableOp>(Op)) {
    if (getResult().getType() != V.getType())
      return emitOpError() << getOperationName()
                           << " result type must match the type of the global"
                              " variable being referenced.";
  } else if (auto F = mlir::dyn_cast_or_null<FunctionOp>(Op)) {
    if (getResult().getType() != F.getFunctionType())
      return emitOpError() << getOperationName()
                           << " result type must match the type of the function"
                              " being referenced.";
  } else {
    return emitOpError() << getOperationName()
                         << " must reference a global variable or function in"
                            " the enclosing 'clift.module' operation.";
  }

  return mlir::success();
}

//===-------------------------------- CallOp ------------------------------===//

static FunctionTypeAttr getFunctionOrFunctionPointerTypeAttr(ValueType Type) {
  ValueType ValueT = decomposeTypedef(Type).Type;
  if (auto P = mlir::dyn_cast<PointerType>(ValueT))
    ValueT = decomposeTypedef(P.getPointeeType()).Type;
  return getFunctionTypeAttr(ValueT);
}

mlir::ParseResult CallOp::parse(OpAsmParser &Parser, OperationState &Result) {
  OpAsmParser::UnresolvedOperand FunctionOperand;
  if (Parser.parseOperand(FunctionOperand).failed())
    return mlir::failure();

  auto ArgumentOperandsLoc = Parser.getCurrentLocation();
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> ArgumentOperands;
  if (Parser.parseLParen().failed())
    return mlir::failure();
  if (Parser.parseOperandList(ArgumentOperands).failed())
    return mlir::failure();
  if (Parser.parseRParen().failed())
    return mlir::failure();

  if (Parser.parseOptionalAttrDict(Result.attributes).failed())
    return mlir::failure();

  if (Parser.parseColon().failed())
    return mlir::failure();

  mlir::SMLoc FunctionTypeLoc = Parser.getCurrentLocation();
  clift::ValueType FunctionType;
  if (Parser.parseType(FunctionType).failed())
    return mlir::failure();

  auto FunctionTypeAttr = getFunctionOrFunctionPointerTypeAttr(FunctionType);
  if (not FunctionTypeAttr)
    return Parser.emitError(FunctionTypeLoc) << "expected Clift function or "
                                                "pointer-to-function type";

  llvm::SmallVector<mlir::Type, 4> ArgumentTypes;

  if (Parser.parseOptionalKeyword("as").succeeded()) {
    if (Parser.parseLParen().failed())
      return mlir::failure();

    if (Parser.parseOptionalRParen().failed()) {
      if (Parser.parseTypeList(ArgumentTypes).failed())
        return mlir::failure();

      if (Parser.parseRParen().failed())
        return mlir::failure();
    }
  } else {
    auto ParameterTypes = FunctionTypeAttr.getArgumentTypes();
    ArgumentTypes.reserve(ParameterTypes.size());

    for (mlir::Type T : ParameterTypes)
      ArgumentTypes.push_back(mlir::cast<clift::ValueType>(T).removeConst());
  }

  Result.addTypes(FunctionTypeAttr.getResultTypes());

  if (Parser.resolveOperand(FunctionOperand, FunctionType, Result.operands)
        .failed())
    return mlir::failure();

  if (Parser
        .resolveOperands(ArgumentOperands,
                         ArgumentTypes,
                         ArgumentOperandsLoc,
                         Result.operands)
        .failed())
    return mlir::failure();

  return mlir::success();
}

void CallOp::print(OpAsmPrinter &Printer) {
  Printer << ' ';
  Printer << getFunction();
  Printer << '(';
  Printer << getArguments();
  Printer << ')';

  Printer.printOptionalAttrDict(getOperation()->getAttrs(), {});
  Printer << ' ' << ':' << ' ';

  auto FunctionType = getFunction().getType();
  Printer << FunctionType;

  auto FunctionTypeAttr = getFunctionOrFunctionPointerTypeAttr(FunctionType);
  revng_assert(FunctionTypeAttr); // Checked by verify.

  auto ArgumentTypes = getArguments().getTypes();
  auto ParameterTypes = FunctionTypeAttr.getArgumentTypes();

  bool RequiresExplicitArgumentTypes = false;
  for (auto &&[ArgumentT, ParameterT] :
       llvm::zip_equal(ArgumentTypes, ParameterTypes)) {
    auto ParameterValueT = mlir::cast<clift::ValueType>(ParameterT);

    if (ArgumentT != ParameterValueT.removeConst()) {
      RequiresExplicitArgumentTypes = true;
      break;
    }
  }

  if (RequiresExplicitArgumentTypes) {
    Printer << ' ' << "as" << ' ' << '(';
    Printer << getArguments().getTypes();
    Printer << ')';
  }
}

mlir::LogicalResult CallOp::verify() {
  auto FunctionType = mlir::cast<clift::ValueType>(getFunction().getType());
  auto FunctionTypeAttr = getFunctionOrFunctionPointerTypeAttr(FunctionType);
  if (not FunctionTypeAttr)
    return emitOpError() << getOperationName()
                         << " function argument must have function or pointer"
                         << "-to-function type.";

  auto ArgumentTypes = getArguments().getTypes();
  auto ParameterTypes = FunctionTypeAttr.getArgumentTypes();

  if (ArgumentTypes.size() != ParameterTypes.size())
    return emitOpError() << getOperationName()
                         << " argument count must match the number of function"
                            " parameters.";

  for (auto &&[ArgumentT, ParameterT] :
       llvm::zip_equal(ArgumentTypes, ParameterTypes)) {
    auto ArgumentValueT = mlir::cast<clift::ValueType>(ArgumentT);
    auto ParameterValueT = mlir::cast<clift::ValueType>(ParameterT);

    if (ArgumentValueT.removeConst() != ParameterValueT.removeConst())
      return emitOpError() << getOperationName()
                           << " argument types must match the parameter types"
                              " of the function, ignoring qualifiers.";
  }

  auto ReturnT = mlir::cast<clift::ValueType>(FunctionTypeAttr.getReturnType());
  auto ResultT = mlir::cast<clift::ValueType>(getResult().getType());

  if (ResultT != ReturnT.removeConst())
    return emitOpError() << getOperationName()
                         << " result type must match the return type of the"
                            " function, ignoring qualifiers.";

  return mlir::success();
}
