//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"

#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/RegionGraphTraits.h"

#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"

namespace mlir {

static ParseResult parseCliftOpTypesImpl(OpAsmParser &Parser,
                                         Type *Result,
                                         llvm::ArrayRef<Type *> Arguments);

static void printCliftOpTypesImpl(OpAsmPrinter &Printer,
                                  Type Result,
                                  llvm::ArrayRef<Type> Arguments);

template<std::same_as<Type>... Ts>
static ParseResult
parseCliftOpTypes(OpAsmParser &Parser, Type &Result, Ts &...Arguments) {
  static_assert(sizeof...(Ts) > 0);
  return parseCliftOpTypesImpl(Parser, &Result, { &Arguments... });
}

template<std::same_as<Type>... Ts>
static ParseResult
parseCliftOpOperandTypes(OpAsmParser &Parser, Ts &...Arguments) {
  static_assert(sizeof...(Ts) > 0);
  return parseCliftOpTypesImpl(Parser, nullptr, { &Arguments... });
}

template<std::same_as<Type>... Ts>
static void printCliftOpTypes(OpAsmPrinter &Printer,
                              Operation *Op,
                              Type Result,
                              Ts... Arguments) {
  static_assert(sizeof...(Ts) > 0);
  printCliftOpTypesImpl(Printer, Result, { Arguments... });
}

template<std::same_as<Type>... Ts>
static void printCliftOpOperandTypes(OpAsmPrinter &Printer,
                                     Operation *Op,
                                     Ts... Arguments) {
  static_assert(sizeof...(Ts) > 0);
  printCliftOpTypesImpl(Printer, nullptr, { Arguments... });
}

static ParseResult parseCliftPointerArithmeticOpTypes(OpAsmParser &Parser,
                                                      Type &Result,
                                                      Type &Lhs,
                                                      Type &Rhs);

static void printCliftPointerArithmeticOpTypes(OpAsmPrinter &Parser,
                                               Operation *Op,
                                               Type Result,
                                               Type Lhs,
                                               Type Rhs);

static ParseResult parseCliftTernaryOpTypes(OpAsmParser &Parser,
                                            Type &Condition,
                                            Type &Lhs,
                                            Type &Rhs);

static void printCliftTernaryOpTypes(OpAsmPrinter &Printer,
                                     Operation *Op,
                                     Type Condition,
                                     Type Lhs,
                                     Type Rhs);

} // namespace mlir

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

bool clift::hasModuleAttr(mlir::ModuleOp Module) {
  llvm::StringRef AttrName = CliftDialect::getModuleAttrName();
  return Module->hasAttrOfType<mlir::UnitAttr>(AttrName);
}

void clift::setModuleAttr(mlir::ModuleOp Module) {
  Module->setAttr(CliftDialect::getModuleAttrName(),
                  mlir::UnitAttr::get(Module.getContext()));
}

YieldOp clift::getExpressionYieldOp(Region &R) {
  if (R.empty())
    return {};

  Block &B = R.front();

  if (B.empty())
    return {};

  return mlir::dyn_cast<clift::YieldOp>(B.back());
}

mlir::Value clift::getExpressionValue(Region &R) {
  if (auto Yield = getExpressionYieldOp(R))
    return Yield.getValue();

  return {};
}

ValueType clift::getExpressionType(Region &R) {
  if (auto Value = getExpressionValue(R))
    return mlir::cast<ValueType>(Value.getType());

  return {};
}

//===-------------------------- Type constraints --------------------------===//

bool clift::impl::verifyPrimitiveTypeOf(ValueType Type, PrimitiveKind Kind) {
  if (auto T = mlir::dyn_cast<PrimitiveType>(Type))
    return T.getKind() == Kind;

  return false;
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
ParseResult mlir::parseCliftOpTypesImpl(OpAsmParser &Parser,
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
void mlir::printCliftOpTypesImpl(OpAsmPrinter &Printer,
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

//===----------------------------- FunctionOp -----------------------------===//

void FunctionOp::build(OpBuilder &Builder,
                       OperationState &State,
                       llvm::StringRef Name,
                       clift::FunctionType FunctionType) {
  size_t ArgumentCount = FunctionType.getArgumentTypes().size();

  llvm::SmallVector<mlir::Attribute> Array;
  Array.resize(std::max<size_t>(ArgumentCount, 1),
               mlir::DictionaryAttr::get(Builder.getContext()));

  auto GetArrayAttr = [&](unsigned Count) {
    return mlir::ArrayAttr::get(Builder.getContext(),
                                llvm::ArrayRef(Array).take_front(Count));
  };

  build(Builder,
        State,
        Name,
        FunctionType,
        /*arg_attrs=*/GetArrayAttr(ArgumentCount),
        /*res_attrs=*/GetArrayAttr(1));
}

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
  clift::ValueType Type;
  if (Parser.parseType(Type).failed())
    return mlir::failure();

  auto FunctionType = mlir::dyn_cast<clift::FunctionType>(Type);
  if (not FunctionType)
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

  if (ResultTypes.empty()) {
    ResultTypes.push_back(PrimitiveType::get(Parser.getContext(),
                                             PrimitiveKind::VoidKind,
                                             0,
                                             false));
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
  Printer.printType(getCliftFunctionType());
  Printer << '>';

  auto FunctionType = getCliftFunctionType();

  function_interface_impl::printFunctionSignature(Printer,
                                                  *this,
                                                  FunctionType
                                                    .getArgumentTypes(),
                                                  /*isVariadic=*/false,
                                                  FunctionType
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

mlir::LogicalResult FunctionOp::verify() {
  auto ReturnType = mlir::cast<ValueType>(getCliftFunctionType()
                                            .getReturnType());

  bool IsVoid = isVoid(ReturnType);
  auto Result = (*this)->walk([&](ReturnOp Op) -> mlir::WalkResult {
    clift::ValueType Type = getExpressionType(Op.getResult());

    if (IsVoid) {
      if (Type)
        return Op->emitOpError() << "cannot return expression in function "
                                    "returning void.";
    } else if (not Type) {
      return Op->emitOpError() << "must return a value in function not "
                                  "returning void.";
    } else if (Type != ReturnType) {
      return Op->emitOpError() << "type does not match the function return "
                                  "type";
    }

    return mlir::success();
  });

  return mlir::failure(Result.wasInterrupted());
}

ArrayRef<Type> FunctionOp::getArgumentTypes() {
  return getCliftFunctionType().getArgumentTypes();
}

ArrayRef<Type> FunctionOp::getResultTypes() {
  return getCliftFunctionType().getResultTypes();
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
    return mlir::failure();

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
  if (mlir::Region &R = getResult(); not R.empty()) {
    auto EmitError = [&]() -> mlir::InFlightDiagnostic {
      return emitOpError() << getOperationName() << " type ";
    };

    if (verifyReturnType(EmitError, getExpressionType(R)).failed())
      return mlir::failure();
  }

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

//===------------------------------ StringOp ------------------------------===//

mlir::LogicalResult StringOp::verify() {
  auto ArrayT = mlir::dyn_cast<ArrayType>(getResult().getType());
  if (not ArrayT or not ArrayT.isConst())
    return emitOpError() << getOperationName()
                         << " result must have const array type.";

  auto CharT = mlir::dyn_cast<PrimitiveType>(ArrayT.getElementType());
  if (not CharT or CharT.getKind() != PrimitiveKind::NumberKind
      or CharT.getSize() != 1)
    return emitOpError() << getOperationName()
                         << " result must have number8_t element type.";

  if (ArrayT.getElementsCount() != getValue().size() + 1)
    return emitOpError() << getOperationName()
                         << " result type length must match string length"
                            " (including null terminator).";

  return mlir::success();
}

//===----------------------- UnaryIntegerMutationOp -----------------------===//

mlir::LogicalResult clift::impl::verifyUnaryIntegerMutationOp(Operation *Op) {
  if (not mlir::clift::isLvalueExpression(Op->getOperand(0)))
    return Op->emitOpError()
           << Op->getName() << " operand must be an lvalue-expression.";

  return mlir::success();
}

//===------------------- Pointer arithmetic expressions -------------------===//

ParseResult mlir::parseCliftPointerArithmeticOpTypes(OpAsmParser &Parser,
                                                     Type &Result,
                                                     Type &Lhs,
                                                     Type &Rhs) {
  SMLoc TypesLoc = Parser.getCurrentLocation();

  if (Parser.parseType(Lhs).failed())
    return mlir::failure();

  if (Parser.parseComma().failed())
    return mlir::failure();

  if (Parser.parseType(Rhs).failed())
    return mlir::failure();

  auto LhsPT = mlir::dyn_cast<PointerType>(dealias(Lhs, true));
  auto RhsPT = mlir::dyn_cast<PointerType>(dealias(Rhs, true));

  if (static_cast<bool>(LhsPT) == static_cast<bool>(RhsPT))
    return Parser.emitError(TypesLoc, "Expected exactly one pointer type.");

  Result = clift::removeConst(LhsPT ? Lhs : Rhs);

  return mlir::success();
}

void mlir::printCliftPointerArithmeticOpTypes(OpAsmPrinter &Printer,
                                              Operation *Op,
                                              Type Result,
                                              Type Lhs,
                                              Type Rhs) {
  Printer << Lhs;
  Printer << ',';
  Printer << Rhs;
}

static mlir::LogicalResult verifyPointerArithmeticOp(mlir::Operation *Op) {
  auto LhsT = mlir::cast<clift::ValueType>(Op->getOperand(0).getType());
  auto RhsT = mlir::cast<clift::ValueType>(Op->getOperand(1).getType());

  auto LhsPT = mlir::dyn_cast<PointerType>(dealias(LhsT, true));
  auto RhsPT = mlir::dyn_cast<PointerType>(dealias(RhsT, true));

  if (static_cast<bool>(LhsPT) == static_cast<bool>(RhsPT))
    return Op->emitOpError() << "requires exactly one pointer operand.";

  auto PointerType = LhsPT ? LhsPT : RhsPT;
  auto IntegerType = mlir::dyn_cast<clift::PrimitiveType>(dealias(LhsPT ? RhsT :
                                                                          LhsT,
                                                                  true));

  if (not IntegerType or not isIntegerKind(IntegerType.getKind()))
    return Op->emitOpError() << "requires an integer operand.";

  if (mlir::isa<PtrSubOp>(Op)) {
    if (not LhsPT)
      return Op->emitOpError() << "left operand must have pointer type.";
  }

  if (IntegerType.getSize() != PointerType.getPointerSize())
    return Op->emitOpError() << "pointer and integer operand sizes must "
                                "match.";

  if (not isObjectType(PointerType.getPointeeType()))
    return Op->emitOpError() << "operand pointee must have object type.";

  if (Op->getResult(0).getType() != PointerType.removeConst())
    return Op->emitOpError() << "result and pointer operand types must match.";

  return mlir::success();
}

unsigned
clift::impl::getPointerArithmeticPointerOperandIndex(mlir::Operation *Op) {
  return isPointerType(Op->getOperand(0).getType()) ? 0 : 1;
}

unsigned
clift::impl::getPointerArithmeticOffsetOperandIndex(mlir::Operation *Op) {
  return isPointerType(Op->getOperand(0).getType()) ? 1 : 0;
}

//===------------------------------ PtrAddOp ------------------------------===//

mlir::LogicalResult PtrAddOp::verify() {
  return verifyPointerArithmeticOp(getOperation());
}

//===------------------------------ PtrSubOp ------------------------------===//

mlir::LogicalResult PtrSubOp::verify() {
  return verifyPointerArithmeticOp(getOperation());
}

//===------------------------------ PtrDiffOp -----------------------------===//

mlir::LogicalResult PtrDiffOp::verify() {
  auto LhsPT = mlir::dyn_cast<PointerType>(dealias(getLhs().getType(), true));
  auto RhsPT = mlir::dyn_cast<PointerType>(dealias(getRhs().getType(), true));

  if (not LhsPT or not RhsPT)
    return emitOpError() << getOperationName()
                         << " requires two pointer operands.";

  auto PointeeType = LhsPT.getPointeeType();
  if (PointeeType.removeConst() != RhsPT.getPointeeType().removeConst())
    return emitOpError() << getOperationName()
                         << " operand pointee types must match, ignoring"
                            " qualifiers.";

  if (not isObjectType(PointeeType))
    return emitOpError() << getOperationName()
                         << " operand pointee must have object type.";

  auto IntegerType = mlir::dyn_cast<PrimitiveType>(getResult().getType());
  if (not IntegerType or IntegerType.getKind() != PrimitiveKind::SignedKind
      or IntegerType.getSize() != LhsPT.getPointerSize())
    return emitOpError() << getOperationName()
                         << " result must have primitive signed integer type"
                            " with size matching that of the operand type.";

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
    if (auto ResUnderlyingT = getUnderlyingIntegerType(ResT)) {
      auto ArgUnderlyingT = getUnderlyingIntegerType(ArgT);
      if (not ArgUnderlyingT)
        return emitOpError() << " argument must have integer type.";

      if (ResUnderlyingT.getKind() != ArgUnderlyingT.getKind())
        return emitOpError() << " result and argument types must be equal in"
                                " kind.";
    } else if (auto ResPointerT = getPointerType(ResT)) {
      auto ArgPointerT = getPointerType(ArgT);
      if (not ArgPointerT)
        return emitOpError() << " argument must have pointer type.";

      if (ResPointerT.getPointeeType() != ArgPointerT.getPointeeType())
        return emitOpError() << " result and argument must have equal pointee "
                                " types.";
    } else {
      return emitOpError() << " result must have integer or pointer type.";
    }

    if (Kind == CastKind::Extend) {
      if (ResT.getByteSize() <= ArgT.getByteSize())
        return emitOpError() << " result type must be wider than the argument"
                                " type.";
    } else {
      if (ResT.getByteSize() >= ArgT.getByteSize())
        return emitOpError() << " result type must be narrower than the"
                                " argument type.";
    }
  } break;
  case CastKind::Bitcast: {
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
    } else if (auto FunctionT = mlir::dyn_cast<FunctionType>(ArgT)) {
      if (PtrT.getPointeeType() != FunctionT)
        return emitOpError() << getOperationName()
                             << " the pointee type of the result type must be"
                                " equal to the argument type.";
    } else {
      return emitOpError() << getOperationName()
                           << " argument must have array or function type.";
    }
  } break;
  case CastKind::Convert: {
    bool ArgIsFloat = isFloatType(ArgT);
    bool ResIsFloat = isFloatType(ResT);

    if (not ArgIsFloat and not isIntegerType(ArgT))
      return emitOpError() << " operand must have floating point or integer"
                              " type";

    if (not ResIsFloat and not isIntegerType(ResT))
      return emitOpError() << " result must have floating point or integer"
                              " type";

    if (not ArgIsFloat and not ResIsFloat)
      return emitOpError() << " requires either the operand or result to have"
                              " floating point type.";

    if (equivalent(ArgT, ResT))
      return emitOpError() << " result type cannot match the operand type.";
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

ClassType AccessOp::getClassType() {
  auto ObjectT = dealias(getValue().getType(), /*IgnoreQualifiers=*/true);

  if (isIndirect()) {
    ObjectT = mlir::cast<PointerType>(ObjectT).getPointeeType();
    ObjectT = dealias(ObjectT, /*IgnoreQualifiers=*/true);
  }

  return mlir::cast<ClassType>(ObjectT.removeConst());
}

FieldAttr AccessOp::getFieldAttr() {
  return getClassType().getFields()[getMemberIndex()];
}

mlir::LogicalResult AccessOp::verify() {
  auto ObjectT = dealias(getValue().getType());

  if (auto PointerT = mlir::dyn_cast<PointerType>(ObjectT)) {
    if (not isIndirect())
      return emitOpError() << getOperationName()
                           << " operand must have pointer type.";

    ObjectT = dealias(PointerT.getPointeeType(), /*IgnoreQualifiers=*/true);
  }

  auto Class = mlir::dyn_cast<ClassType>(ObjectT);
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
  auto Module = getOperation()->getParentOfType<mlir::ModuleOp>();
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
                            " the enclosing 'builtin.module' operation.";
  }

  return mlir::success();
}

//===-------------------------------- CallOp ------------------------------===//

namespace {

using DefaultArgumentTypeProvider = //
  llvm::function_ref<clift::ValueType(unsigned)>;

/// Parses an argument list delimited by parentheses with optional operand
/// types. After parsing, default operand types may be provided.
///
/// Syntax examples:
///   (%0)
///   (%0, %1)
///   (%0 : !int32_t, %1)
///   (%0 : !int32_t, %1 : !int32_t)
class ArgumentListParser {
public:
  ParseResult parse(OpAsmParser &Parser, bool RequireTypes) {
    Location = Parser.getCurrentLocation();

    if (Parser.parseLParen().failed())
      return mlir::failure();

    if (Parser.parseOptionalRParen().failed()) {
      do {
        if (Parser.parseOperand(Operands.emplace_back()).failed())
          return mlir::failure();

        mlir::Type Type = {};
        if (Parser.parseOptionalColon().succeeded()) {
          if (Parser.parseType(Type).failed())
            return mlir::failure();
        } else if (RequireTypes) {
          // Parsing an optional colon already failed, but it was actually
          // required. The easiest way to produce the appropriate error message
          // is to try parsing a non-optional colon again.
          return Parser.parseColon();
        }

        Types.push_back(Type);
      } while (Parser.parseOptionalComma().succeeded());

      if (Parser.parseRParen().failed())
        return mlir::failure();
    }

    return mlir::success();
  }

  ParseResult resolveOperands(OpAsmParser &Parser, OperationState &Result) {
    return Parser.resolveOperands(Operands, Types, Location, Result.operands);
  }

  ParseResult resolveOperands(OpAsmParser &Parser,
                              OperationState &Result,
                              DefaultArgumentTypeProvider GetDefaultType) {
    for (auto [I, T] : llvm::enumerate(Types)) {
      if (not T) {
        if (clift::ValueType DefaultType = GetDefaultType(I))
          T = DefaultType.removeConst();
      }
    }

    return resolveOperands(Parser, Result);
  }

private:
  SMLoc Location;
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> Operands;
  llvm::SmallVector<mlir::Type> Types;
};

} // namespace

static void printArgumentList(OpAsmPrinter &Printer,
                              mlir::OperandRange Operands,
                              DefaultArgumentTypeProvider GetDefaultType) {
  Printer << '(';
  for (auto [I, V] : llvm::enumerate(Operands)) {
    if (I != 0)
      Printer << ", ";

    Printer << V;
    if (clift::ValueType DefaultType = GetDefaultType(I))
      if (V.getType() != DefaultType.removeConst())
        Printer << " : " << V.getType();
  }
  Printer << ')';
}

static auto makeCallArgumentTypeAccessor(clift::FunctionType Function) {
  return [Function](unsigned I) -> clift::ValueType {
    auto ParameterTypes = Function.getArgumentTypes();
    return I < ParameterTypes.size() ?
             mlir::cast<clift::ValueType>(ParameterTypes[I]) :
             clift::ValueType();
  };
}

mlir::ParseResult CallOp::parse(OpAsmParser &Parser, OperationState &Result) {
  OpAsmParser::UnresolvedOperand FunctionOperand;
  if (Parser.parseOperand(FunctionOperand).failed())
    return mlir::failure();

  ArgumentListParser Arguments;
  if (Arguments.parse(Parser, /*RequireTypes=*/false).failed())
    return mlir::failure();

  if (Parser.parseOptionalAttrDict(Result.attributes).failed())
    return mlir::failure();

  if (Parser.parseColon().failed())
    return mlir::failure();

  mlir::SMLoc FunctionTypeLoc = Parser.getCurrentLocation();
  clift::ValueType FunctionValueType;
  if (Parser.parseType(FunctionValueType).failed())
    return mlir::failure();

  auto
    FunctionType = getFunctionOrFunctionPointerFunctionType(FunctionValueType);

  if (not FunctionType)
    return Parser.emitError(FunctionTypeLoc) << "expected Clift function or "
                                                "pointer-to-function type";

  Result.addTypes(FunctionType.getResultTypes());

  if (Parser.resolveOperand(FunctionOperand, FunctionValueType, Result.operands)
        .failed())
    return mlir::failure();

  if (Arguments
        .resolveOperands(Parser,
                         Result,
                         makeCallArgumentTypeAccessor(FunctionType))
        .failed())
    return mlir::failure();

  return mlir::success();
}

void CallOp::print(OpAsmPrinter &Printer) {
  auto Type = getFunction().getType();
  auto FunctionType = getFunctionOrFunctionPointerFunctionType(Type);
  revng_assert(FunctionType); // Checked by verify.

  Printer << ' ';
  Printer << getFunction();
  printArgumentList(Printer,
                    getArguments(),
                    makeCallArgumentTypeAccessor(FunctionType));

  Printer.printOptionalAttrDict(getOperation()->getAttrs(), {});
  Printer << ' ' << ':' << ' ' << Type;
}

mlir::LogicalResult CallOp::verify() {
  auto FunctionValueType = mlir::cast<clift::ValueType>(getFunction()
                                                          .getType());
  auto
    FunctionType = getFunctionOrFunctionPointerFunctionType(FunctionValueType);
  if (not FunctionType)
    return emitOpError() << getOperationName()
                         << " function argument must have function or pointer"
                         << "-to-function type.";

  auto ArgumentTypes = getArguments().getTypes();
  auto ParameterTypes = FunctionType.getArgumentTypes();

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

  auto ReturnT = mlir::cast<clift::ValueType>(FunctionType.getReturnType());
  auto ResultT = mlir::cast<clift::ValueType>(getResult().getType());

  if (ResultT != ReturnT.removeConst())
    return emitOpError() << getOperationName()
                         << " result type must match the return type of the"
                            " function, ignoring qualifiers.";

  return mlir::success();
}

//===------------------------------ TernaryOp -----------------------------===//

ParseResult mlir::parseCliftTernaryOpTypes(OpAsmParser &Parser,
                                           Type &Condition,
                                           Type &Lhs,
                                           Type &Rhs) {
  if (Parser.parseType(Condition).failed())
    return mlir::failure();

  if (Parser.parseComma().failed())
    return mlir::failure();

  if (Parser.parseType(Lhs).failed())
    return mlir::failure();

  if (Parser.parseOptionalComma().succeeded()) {
    if (Parser.parseType(Rhs).failed())
      return mlir::failure();
  } else {
    Rhs = Lhs;
  }

  return mlir::success();
}

void mlir::printCliftTernaryOpTypes(OpAsmPrinter &Printer,
                                    Operation *Op,
                                    Type Condition,
                                    Type Lhs,
                                    Type Rhs) {
  Printer << Condition;
  Printer << ',';
  Printer << Lhs;

  if (Lhs != Rhs) {
    Printer << ',';
    Printer << Rhs;
  }
}

//===----------------------------- AggregateOp ----------------------------===//

static auto makeAggregateArgumentTypeAccessor(clift::ValueType Type) {
  auto UnderlyingType = dealias(Type, /*IgnoreQualifiers=*/true);
  return [UnderlyingType](unsigned I) -> clift::ValueType {
    if (auto Array = mlir::dyn_cast<ArrayType>(UnderlyingType))
      return Array.getElementType();

    if (auto Struct = mlir::dyn_cast<StructType>(UnderlyingType)) {
      auto Fields = Struct.getFields();
      return I < Fields.size() ? Fields[I].getType() : clift::ValueType();
    }

    return {};
  };
}

mlir::ParseResult AggregateOp::parse(OpAsmParser &Parser,
                                     OperationState &Result) {
  ArgumentListParser Arguments;
  if (Arguments.parse(Parser, /*Requiretypes=*/false).failed())
    return mlir::failure();

  if (Parser.parseOptionalAttrDict(Result.attributes).failed())
    return mlir::failure();

  if (Parser.parseColon().failed())
    return mlir::failure();

  clift::ValueType ResultType;
  if (Parser.parseType(ResultType).failed())
    return mlir::failure();

  if (Arguments
        .resolveOperands(Parser,
                         Result,
                         makeAggregateArgumentTypeAccessor(ResultType))
        .failed())
    return mlir::failure();

  Result.addTypes({ ResultType });

  return mlir::success();
}

void AggregateOp::print(OpAsmPrinter &Printer) {
  clift::ValueType ResultType = getResult().getType();

  printArgumentList(Printer,
                    getInitializers(),
                    makeAggregateArgumentTypeAccessor(ResultType));

  Printer.printOptionalAttrDict(getOperation()->getAttrs(), {});

  Printer << " : ";
  Printer << ResultType;
}

mlir::LogicalResult AggregateOp::verify() {
  auto InitializerTypes = getInitializers().getTypes();
  auto AT = dealias(getResult().getType(), /*IgnoreQualifiers=*/true);

  if (auto T = mlir::dyn_cast<StructType>(AT)) {
    auto Fields = T.getFields();

    if (InitializerTypes.size() != Fields.size())
      return emitOpError() << getOperationName()
                           << " must initialize all struct members.";

    for (auto [IT, SF] : llvm::zip(InitializerTypes, Fields)) {
      if (not clift::equivalent(IT, SF.getType()))
        return emitOpError() << getOperationName()
                             << " initializer types must match the struct field"
                                " types.";
    }
  } else if (auto T = mlir::dyn_cast<ArrayType>(AT)) {
    if (InitializerTypes.size() != T.getElementsCount())
      return emitOpError() << getOperationName()
                           << " must initialize all array elements.";

    for (auto IT : InitializerTypes) {
      if (not clift::equivalent(IT, T.getElementType()))
        return emitOpError() << getOperationName()
                             << " initializer types must match the array"
                                " element type.";
    }
  } else {
    return emitOpError() << getOperationName()
                         << " result have struct or array type.";
  }

  return mlir::success();
}
