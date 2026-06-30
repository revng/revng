//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/RegionGraphTraits.h"

#include "revng/Clift/Clift.h"
#include "revng/Clift/CliftAttributes.h"
#include "revng/Clift/CliftOpHelpers.h"

std::unique_ptr<mlir::MLIRContext> clift::makeContext() {
  static constexpr auto Threading = mlir::MLIRContext::Threading::DISABLED;
  static const mlir::DialectRegistry Registry = []() -> mlir::DialectRegistry {
    mlir::DialectRegistry Registry;
    Registry.insert<clift::CliftDialect>();
    return Registry;
  }();

  auto Result = std::make_unique<mlir::MLIRContext>(Registry, Threading);
  Result->loadDialect<CliftDialect>();
  return Result;
}

mlir::OwningOpRef<mlir::ModuleOp>
clift::makeModule(mlir::MLIRContext &Context) {
  auto DebugLocation = mlir::UnknownLoc::get(&Context);
  mlir::OwningOpRef<mlir::ModuleOp>
    Result = mlir::ModuleOp::create(DebugLocation);

  Result.get()->setAttr(CliftDialect::getModuleAttrName(),
                        mlir::UnitAttr::get(&Context));

  return Result;
}

using UnresolvedOperandsVector = //
  llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand>;

static mlir::ParseResult parseCliftLoopLabels(mlir::OpAsmParser &Parser,
                                              mlir::IntegerAttr &LabelMask,
                                              UnresolvedOperandsVector &Labels);

static void printCliftLoopLabels(mlir::OpAsmPrinter &Printer,
                                 mlir::Operation *Op,
                                 mlir::IntegerAttr LabelMask,
                                 mlir::OperandRange Labels);

static mlir::ParseResult
parseCliftOpTypesImpl(mlir::OpAsmParser &Parser,
                      mlir::Type *Result,
                      llvm::ArrayRef<mlir::Type *> Arguments);

static void printCliftOpTypesImpl(mlir::OpAsmPrinter &Printer,
                                  mlir::Type Result,
                                  llvm::ArrayRef<mlir::Type> Arguments);

template<std::same_as<mlir::Type>... Ts>
static mlir::ParseResult parseCliftOpTypes(mlir::OpAsmParser &Parser,
                                           mlir::Type &Result,
                                           Ts &...Arguments) {
  static_assert(sizeof...(Ts) > 0);
  return parseCliftOpTypesImpl(Parser, &Result, { &Arguments... });
}

template<std::same_as<mlir::Type>... Ts>
static mlir::ParseResult
parseCliftOpOperandTypes(mlir::OpAsmParser &Parser, Ts &...Arguments) {
  static_assert(sizeof...(Ts) > 0);
  return parseCliftOpTypesImpl(Parser, nullptr, { &Arguments... });
}

template<std::same_as<mlir::Type>... Ts>
static void printCliftOpTypes(mlir::OpAsmPrinter &Printer,
                              mlir::Operation *Op,
                              mlir::Type Result,
                              Ts... Arguments) {
  static_assert(sizeof...(Ts) > 0);
  printCliftOpTypesImpl(Printer, Result, { Arguments... });
}

template<std::same_as<mlir::Type>... Ts>
static void printCliftOpOperandTypes(mlir::OpAsmPrinter &Printer,
                                     mlir::Operation *Op,
                                     Ts... Arguments) {
  static_assert(sizeof...(Ts) > 0);
  printCliftOpTypesImpl(Printer, nullptr, { Arguments... });
}

static mlir::ParseResult
parseCliftPointerArithmeticOpTypes(mlir::OpAsmParser &Parser,
                                   mlir::Type &Result,
                                   mlir::Type &Lhs,
                                   mlir::Type &Rhs);

static void printCliftPointerArithmeticOpTypes(mlir::OpAsmPrinter &Parser,
                                               mlir::Operation *Op,
                                               mlir::Type Result,
                                               mlir::Type Lhs,
                                               mlir::Type Rhs);

static mlir::ParseResult parseCliftTernaryOpTypes(mlir::OpAsmParser &Parser,
                                                  mlir::Type &Condition,
                                                  mlir::Type &Lhs,
                                                  mlir::Type &Rhs);

static void printCliftTernaryOpTypes(mlir::OpAsmPrinter &Printer,
                                     mlir::Operation *Op,
                                     mlir::Type Condition,
                                     mlir::Type Lhs,
                                     mlir::Type Rhs);

#define GET_OP_CLASSES
#include "revng/Clift/Clift.cpp.inc"

namespace func_impl = mlir::function_interface_impl;

using namespace clift;

void CliftDialect::registerOperations() {
  addOperations</* Include the auto-generated clift operations */
#define GET_OP_LIST
#include "revng/Clift/Clift.cpp.inc"
                /* End of operations list */>();
}

bool clift::isCliftModule(mlir::ModuleOp Module) {
  llvm::StringRef AttrName = CliftDialect::getModuleAttrName();
  return Module->hasAttrOfType<mlir::UnitAttr>(AttrName);
}

const CDataModel &clift::getDataModel(mlir::ModuleOp Module) {
  if (auto Attr = Module->getAttr(CliftDialect::getDataModelAttrName()))
    return mlir::cast<DataModelAttr>(Attr).getDataModel();

  if (auto Dialect = Module->getContext()->getLoadedDialect<CliftDialect>()) {
    if (const auto *DM = Dialect->getDefaultDataModel())
      return *DM;
  }

  revng_abort("The module does not specify a data model.");
}

const CDataModel &clift::getDataModel(FunctionOp Function) {
  auto Module = Function->getParentOfType<mlir::ModuleOp>();
  revng_assert(Module, "The function must be contained within a module.");
  return getDataModel(Module);
}

void clift::setDataModel(mlir::ModuleOp Module, const CDataModel &DataModel) {
  Module->setAttr(CliftDialect::getDataModelAttrName(),
                  DataModelAttr::get(Module.getContext(), DataModel));
}

YieldOp clift::getExpressionYieldOp(mlir::Region &R) {
  if (R.empty())
    return {};

  mlir::Block &B = R.front();

  if (B.empty())
    return {};

  return mlir::dyn_cast<clift::YieldOp>(B.back());
}

mlir::Value clift::getExpressionValue(mlir::Region &R) {
  if (auto Yield = getExpressionYieldOp(R))
    return Yield.getValue();

  return {};
}

mlir::Type clift::getExpressionType(mlir::Region &R) {
  if (auto Value = getExpressionValue(R))
    return Value.getType();

  return {};
}

//===---------------------------- Region types ----------------------------===//

template<typename OpInterface>
static bool verifyRegionContent(mlir::Region &R, const bool Required) {
  if (R.empty())
    return not Required;

  if (not R.hasOneBlock())
    return false;

  for (mlir::Operation &Op : R.front()) {
    if (not mlir::isa<OpInterface>(&Op))
      return false;
  }

  return true;
}

bool clift::impl::verifyStatementRegion(mlir::Region &R) {
  return verifyRegionContent<StatementOpInterface>(R, false);
}

bool clift::impl::verifyExpressionRegion(mlir::Region &R, const bool Required) {
  if (not verifyRegionContent<ExpressionOpInterface>(R, Required))
    return false;

  return R.empty() or static_cast<bool>(getExpressionYieldOp(R));
}

//===-------------------------- Operation parsing -------------------------===//

template<typename TypeOrPointer>
static mlir::Type deduceResultType(llvm::ArrayRef<TypeOrPointer> Arguments) {
  const auto getType = [](TypeOrPointer Argument) -> mlir::Type {
    if constexpr (std::is_same_v<TypeOrPointer, mlir::Type>) {
      return Argument;
    } else {
      return *Argument;
    }
  };

  auto CommonType = removeConst(getType(Arguments.front()));
  for (TypeOrPointer Argument : Arguments.slice(0)) {
    if (removeConst(getType(Argument)) != CommonType)
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
static mlir::ParseResult
parseCliftOpTypesImpl(mlir::OpAsmParser &Parser,
                      mlir::Type *Result,
                      llvm::ArrayRef<mlir::Type *> Arguments) {
  mlir::Type &FirstArgument = *Arguments.front();
  if (Parser.parseOptionalLParen().succeeded()) {
    if (Parser.parseType(FirstArgument).failed())
      return mlir::failure();

    for (mlir::Type *Argument : Arguments.slice(1)) {
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

    for (mlir::Type *Argument : Arguments.slice(1))
      *Argument = FirstArgument;
  }

  if (Result != nullptr) {
    if (Parser.parseOptionalArrow().succeeded()) {
      if (Parser.parseType(*Result).failed())
        return mlir::failure();
    } else if (mlir::Type Deduced = deduceResultType(Arguments)) {
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
static void printCliftOpTypesImpl(mlir::OpAsmPrinter &Printer,
                                  mlir::Type Result,
                                  llvm::ArrayRef<mlir::Type> Arguments) {
  bool ArgumentsEqual = llvm::all_equal(Arguments);
  mlir::Type FirstArgument = Arguments.front();

  if (ArgumentsEqual) {
    Printer << FirstArgument;
  } else {
    Printer << "(";
    Printer << FirstArgument;
    for (mlir::Type Argument : Arguments.slice(1)) {
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
        IsDeducible = Result == removeConst(FirstArgument);
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

void FunctionOp::build(mlir::OpBuilder &Builder,
                       mlir::OperationState &State,
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

mlir::ParseResult FunctionOp::parse(mlir::OpAsmParser &Parser,
                                    mlir::OperationState &Result) {
  mlir::StringAttr SymbolNameAttr;
  if (Parser
        .parseSymbolName(SymbolNameAttr,
                         mlir::SymbolTable::getSymbolAttrName(),
                         Result.attributes)
        .failed())
    return mlir::failure();

  if (Parser.parseLess().failed())
    return mlir::failure();

  auto FunctionTypeLoc = Parser.getCurrentLocation();
  mlir::Type Type;
  if (Parser.parseType(Type).failed())
    return mlir::failure();

  auto FunctionType = mlir::dyn_cast<clift::FunctionType>(Type);
  if (not FunctionType)
    return Parser.emitError(FunctionTypeLoc) << "expected Clift function or "
                                                "pointer-to-function type.";

  if (Parser.parseGreater().failed())
    return mlir::failure();

  llvm::SmallVector<mlir::OpAsmParser::Argument> Arguments;
  llvm::SmallVector<mlir::Type> ResultTypes;
  llvm::SmallVector<mlir::DictionaryAttr> ResultAttrs;
  bool IsVariadic = false;

  auto RoughResultTypeLocation = Parser.getCurrentLocation();
  if (func_impl::parseFunctionSignature(Parser,
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
    ResultTypes.push_back(VoidType::get(Parser.getContext()));
    ResultAttrs.push_back(mlir::DictionaryAttr::get(Parser.getContext()));
  }

  llvm::SmallVector<mlir::Type> ArgumentTypes;
  for (auto &Argument : Arguments)
    ArgumentTypes.push_back(Argument.type);

  Result.addAttribute(getFunctionTypeAttrName(Result.name),
                      mlir::TypeAttr::get(FunctionType));

  if (Parser.parseOptionalAttrDictWithKeyword(Result.attributes).failed())
    return mlir::failure();

  func_impl::addArgAndResultAttrs(Parser.getBuilder(),
                                  Result,
                                  Arguments,
                                  ResultAttrs,
                                  getArgAttrsAttrName(Result.name),
                                  getResAttrsAttrName(Result.name));

  auto *Body = Result.addRegion();
  auto RegionParseResult = Parser.parseOptionalRegion(*Body, Arguments);
  if (RegionParseResult.has_value() && mlir::failed(*RegionParseResult))
    return mlir::failure();

  return mlir::success();
}

void FunctionOp::print(mlir::OpAsmPrinter &Printer) {
  auto FunctionType = getFunctionType();

  Printer << ' ';
  Printer.printSymbolName(getSymName());
  Printer << '<';
  Printer.printType(FunctionType);
  Printer << '>';

  func_impl::printFunctionSignature(Printer,
                                    *this,
                                    FunctionType.getArgumentTypes(),
                                    /*isVariadic=*/false,
                                    FunctionType.getResultTypes());

  func_impl::printFunctionAttributes(Printer,
                                     *this,
                                     { getFunctionTypeAttrName(),
                                       getArgAttrsAttrName(),
                                       getResAttrsAttrName() });

  if (mlir::Region &Body = getBody(); !Body.empty()) {
    Printer << ' ';
    Printer.printRegion(Body,
                        /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }
}

llvm::ArrayRef<mlir::Type> FunctionOp::getArgumentTypes() {
  return getFunctionType().getArgumentTypes();
}

llvm::ArrayRef<mlir::Type> FunctionOp::getResultTypes() {
  return getFunctionType().getResultTypes();
}

mlir::Type FunctionOp::cloneTypeWith(mlir::TypeRange Inputs,
                                     mlir::TypeRange Results) {
  revng_abort("Operation not supported");
}

//===-------------------------- GlobalVariableOp --------------------------===//

mlir::LogicalResult GlobalVariableOp::verify() {
  if (mlir::Region &R = getInitializer(); not R.empty()) {
    if (getExpressionType(R) != getType())
      return emitOpError() << getOperationName()
                           << " initializer type must match the variable type";
  }

  return mlir::success();
}

//===----------------------------- Statements -----------------------------===//

static mlir::IntegerAttr makeLoopLabelMask(mlir::MLIRContext *Context,
                                           unsigned Mask) {
  return mlir::IntegerAttr::get(Context, llvm::APSInt(llvm::APInt(2, Mask)));
}

/// If \p OtherLoop is non-null, assigned label operands are copied from it.
/// This is useful for any transforms which change the type of a loop, and must
/// preserve the assigned labels.
static void buildLoop(mlir::OpBuilder &Builder,
                      mlir::OperationState &State,
                      unsigned RegionCount,
                      LoopOpInterface OtherLoop = {}) {
  if (OtherLoop) {
    State.addOperands(OtherLoop->getOperands());
    State.addAttribute(clift::impl::LoopLabelMaskAttrName,
                       OtherLoop->getAttr(clift::impl::LoopLabelMaskAttrName));
  } else {
    State.addAttribute(clift::impl::LoopLabelMaskAttrName,
                       makeLoopLabelMask(Builder.getContext(), 0));
  }

  for (unsigned I = 0; I < RegionCount; ++I)
    State.addRegion();
}

static mlir::ParseResult
parseCliftLoopLabels(mlir::OpAsmParser &Parser,
                     mlir::IntegerAttr &LabelMask,
                     UnresolvedOperandsVector &Labels) {
  unsigned Mask = 0;

  if (Parser.parseOptionalKeyword("break").succeeded()) {
    mlir::OpAsmParser::UnresolvedOperand Operand;
    if (Parser.parseOperand(Operand).failed())
      return mlir::failure();
    Labels.push_back(Operand);
    Mask |= clift::impl::BreakLabelFlag;
  }

  if (Parser.parseOptionalKeyword("continue").succeeded()) {
    mlir::OpAsmParser::UnresolvedOperand Operand;
    if (Parser.parseOperand(Operand).failed())
      return mlir::failure();
    Labels.push_back(Operand);
    Mask |= clift::impl::ContinueLabelFlag;
  }

  LabelMask = makeLoopLabelMask(Parser.getContext(), Mask);

  return mlir::success();
}

static void printCliftLoopLabels(mlir::OpAsmPrinter &Printer,
                                 mlir::Operation *Op,
                                 mlir::IntegerAttr LabelMask,
                                 mlir::OperandRange Labels) {
  unsigned Mask = LabelMask.getValue().getZExtValue();

  unsigned Next = 0;

  if (Mask & clift::impl::BreakLabelFlag) {
    Printer << "break ";
    Printer.printOperand(Labels[Next++]);
  }

  if (Mask & clift::impl::ContinueLabelFlag) {
    if (Next != 0)
      Printer << " ";

    Printer << "continue ";
    Printer.printOperand(Labels[Next++]);
  }
}

//===---------------------------- AssignLabelOp ---------------------------===//

MakeLabelOp AssignLabelOp::getLabelOp() {
  return getLabel().getDefiningOp<MakeLabelOp>();
}

//===-------------------------- BlockStatementOp --------------------------===//

bool BlockStatementOp::isIndirectlyNoFallthrough() {
  return clift::isIndirectlyNoFallthrough(getBlock());
}

//===------------------------------ BreakToOp -----------------------------===//

mlir::LogicalResult BreakToOp::verify() {
  mlir::Operation *Assignment = getLabelAssignmentOp();
  auto Loop = mlir::dyn_cast<LoopOpInterface>(Assignment);

  if (not Loop or getLabel() != Loop.getBreakLabel()) {
    return emitOpError() << getOperationName()
                         << " must target a loop break label.";
  }

  if (not Loop->isAncestor(getOperation())) {
    return emitOpError() << getOperationName()
                         << " must target a nesting loop label.";
  }

  return mlir::success();
}

//===---------------------------- ContinueToOp ----------------------------===//

mlir::LogicalResult ContinueToOp::verify() {
  mlir::Operation *Assignment = getLabelAssignmentOp();
  auto Loop = mlir::dyn_cast<LoopOpInterface>(Assignment);

  if (not Loop or getLabel() != Loop.getContinueLabel()) {
    return emitOpError() << getOperationName()
                         << " must target a loop continue label.";
  }

  if (not Loop->isAncestor(getOperation())) {
    return emitOpError() << getOperationName()
                         << " must target a nesting loop label.";
  }

  return mlir::success();
}

//===------------------------------ DoWhileOp -----------------------------===//

void DoWhileOp::build(mlir::OpBuilder &Builder,
                      mlir::OperationState &State,
                      LoopOpInterface OtherLoop) {
  buildLoop(Builder, State, 2, OtherLoop);
}

mlir::LogicalResult DoWhileOp::verify() {
  if (not isScalarType(getExpressionType(getCondition())))
    return emitOpError() << getOperationName()
                         << " condition requires a scalar type.";

  return mlir::success();
}

//===-------------------------------- ForOp -------------------------------===//

mlir::Value ForOp::getBlockArgumentVariable(mlir::BlockArgument Argument) {
  return getOnlyOp<LocalVariableOp>(getInitializer());
}

bool ForOp::isDiscardedExpression(mlir::Region &R) {
  if (&R == &getInitializer())
    return not getOnlyOp<LocalVariableOp>(R);

  return &R == &getExpression();
}

void ForOp::build(mlir::OpBuilder &Builder,
                  mlir::OperationState &State,
                  LoopOpInterface OtherLoop) {
  buildLoop(Builder, State, 4, OtherLoop);
}

mlir::ParseResult ForOp::parse(mlir::OpAsmParser &Parser,
                               mlir::OperationState &Result) {
  mlir::IntegerAttr LabelMaskAttr;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> LabelOperands;
  mlir::SMLoc LabelOperandsLoc = Parser.getCurrentLocation();

  if (parseCliftLoopLabels(Parser, LabelMaskAttr, LabelOperands))
    return mlir::failure();

  mlir::Type InitType = {};

  auto InitRegion = std::make_unique<mlir::Region>();
  if (Parser.parseOptionalKeyword("init").succeeded()) {
    if (Parser.parseOptionalColon().succeeded()) {
      if (Parser.parseType(InitType).failed())
        return mlir::failure();
    }

    if (Parser.parseRegion(*InitRegion).failed())
      return mlir::failure();
  }

  auto ParseRegion = [&Parser,
                      &InitType](mlir::Region &R) -> mlir::LogicalResult {
    llvm::SmallVector<mlir::OpAsmParser::Argument, 1> Arguments;

    mlir::SMLoc OperandLoc = Parser.getCurrentLocation();
    mlir::OpAsmParser::UnresolvedOperand Operand;

    if (Parser.parseOptionalLParen().succeeded()) {
      if (Parser.parseOperand(Operand).failed())
        return mlir::failure();
      if (Parser.parseRParen().failed())
        return mlir::failure();

      if (not InitType) {
        return Parser.emitError(OperandLoc,
                                "Region operand requires specifying the type "
                                "of the variable declared in the init region "
                                "of this operation.");
      }

      Arguments.emplace_back(Operand, InitType);
    }

    if (Parser.parseRegion(R, Arguments).failed())
      return mlir::failure();

    return mlir::success();
  };

  auto CondRegion = std::make_unique<mlir::Region>();
  if (Parser.parseOptionalKeyword("cond").succeeded()) {
    if (ParseRegion(*CondRegion).failed())
      return mlir::failure();
  }

  auto NextRegion = std::make_unique<mlir::Region>();
  if (Parser.parseOptionalKeyword("next").succeeded()) {
    if (ParseRegion(*NextRegion).failed())
      return mlir::failure();
  }

  if (Parser.parseKeyword("body").failed())
    return mlir::failure();

  auto BodyRegion = std::make_unique<mlir::Region>();
  if (ParseRegion(*BodyRegion).failed())
    return mlir::failure();

  if (Parser.parseOptionalAttrDictWithKeyword(Result.attributes))
    return mlir::failure();

  if (Parser
        .resolveOperands(LabelOperands,
                         LabelType::get(Parser.getContext()),
                         LabelOperandsLoc,
                         Result.operands)
        .failed())
    return mlir::failure();

  Result.addAttribute(clift::impl::LoopLabelMaskAttrName, LabelMaskAttr);
  Result.addRegion(std::move(InitRegion));
  Result.addRegion(std::move(CondRegion));
  Result.addRegion(std::move(NextRegion));
  Result.addRegion(std::move(BodyRegion));

  return mlir::success();
}

void ForOp::print(mlir::OpAsmPrinter &Printer) {
  mlir::Type InitType = {};

  auto SetInitType = [&InitType](mlir::Region &R) {
    if (not InitType and R.getNumArguments() != 0)
      InitType = R.getArgument(0).getType();
  };

  SetInitType(getCondition());
  SetInitType(getExpression());
  SetInitType(getBody());

  Printer << ' ';
  printCliftLoopLabels(Printer, *this, getLabelMaskAttr(), getLabels());
  Printer << ' ';

  if (InitType or not getInitializer().empty()) {
    Printer << " init ";

    if (InitType) {
      Printer << ':';
      Printer << ' ';
      Printer.printType(InitType);
      Printer << ' ';
    }

    Printer.printRegion(getInitializer());
  }

  auto PrintRegion = [&Printer](mlir::Region &R) {
    if (R.getNumArguments() != 0) {
      Printer << '(';
      Printer.printOperand(R.getArgument(0));
      Printer << ')';
      Printer << ' ';
    }

    Printer.printRegion(R, /*printEntryBlockArgs=*/false);
  };

  if (not getCondition().empty()) {
    Printer << " cond ";
    PrintRegion(getCondition());
  }

  if (not getExpression().empty()) {
    Printer << " next ";
    PrintRegion(getExpression());
  }

  Printer << " body ";
  PrintRegion(getBody());

  Printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           clift::impl::LoopLabelMaskAttrName);
}

mlir::LogicalResult ForOp::verify() {
  mlir::Region &Initializer = getInitializer();

  mlir::Type InitType = {};
  if (not Initializer.empty()) {
    mlir::Operation *Op = getOnlyOp(Initializer);

    if (not Op or not mlir::isa<ExpressionStatementOp, LocalVariableOp>(Op)) {
      return emitOpError() << getOperationName()
                           << " initializer region must be empty or contain"
                              " exactly one expression statement or local"
                              " variable declaration.";
    }

    if (auto Local = mlir::dyn_cast<LocalVariableOp>(Op))
      InitType = Local.getType();
  }

  auto CheckRegionArguments =
    [this, &InitType](llvm::StringRef Name,
                      mlir::Region &R) -> mlir::LogicalResult {
    if (R.getNumArguments() != 0) {
      if (R.getNumArguments() != 1) {
        return emitOpError() << getOperationName() << " " << Name
                             << " region may have no more than one argument.";
      }

      if (R.getArgument(0).getType() != InitType) {
        return emitOpError() << getOperationName() << " " << Name
                             << " region argument type must match the type of"
                                " the local variable declaration contained in "
                                " the init region.";
      }
    }

    return mlir::success();
  };

  if (CheckRegionArguments("condition", getCondition()).failed())
    return mlir::failure();
  if (CheckRegionArguments("expression", getExpression()).failed())
    return mlir::failure();
  if (CheckRegionArguments("body", getBody()).failed())
    return mlir::failure();

  if (auto ConditionType = getExpressionType(getCondition())) {
    if (not isScalarType(ConditionType))
      return emitOpError() << getOperationName()
                           << " condition requires a scalar type.";
  }

  return mlir::success();
}

static clift::ExpressionRegionOpInterface
getInitializerExpressionRegions(ForOp Op) {
  using ERI = clift::ExpressionRegionOpInterface;
  return clift::getOnlyOp<ERI>(Op.getInitializer());
}

unsigned ForOp::getExpressionRegionCount() {
  auto Initializer = getInitializerExpressionRegions(*this);
  return 2 + (Initializer ? Initializer.getExpressionRegionCount() : 0);
}

mlir::Region &ForOp::getExpressionRegion(unsigned Index) {
  if (auto Initializer = getInitializerExpressionRegions(*this)) {
    unsigned InitializerCount = Initializer.getExpressionRegionCount();
    if (Index < InitializerCount)
      return Initializer.getExpressionRegion(Index);
    Index -= InitializerCount;
  }
  revng_assert(Index < 2);
  return Index == 0 ? getCondition() : getExpression();
}

//===------------------------------- GotoOp -------------------------------===//

MakeLabelOp GotoOp::getLabelOp() {
  return getLabel().getDefiningOp<MakeLabelOp>();
}

mlir::LogicalResult GotoOp::verify() {
  mlir::Operation *Assignment = getLabelAssignmentOp();

  if (mlir::isa<LoopOpInterface>(Assignment)) {
    if (Assignment->isAncestor(getOperation()))
      return emitOpError() << getOperationName()
                           << " may not target a nesting loop label.";
  }

  return mlir::success();
}

//===-------------------------------- IfOp --------------------------------===//

static bool isIndirectlyNoFallthroughImpl(BranchOpInterface Branch) {
  for (mlir::Region &R : Branch.getBranchRegions()) {
    if (not clift::isIndirectlyNoFallthrough(R))
      return false;
  }
  return true;
}

bool IfOp::isIndirectlyNoFallthrough() const {
  return isIndirectlyNoFallthroughImpl(*this);
}

mlir::LogicalResult IfOp::verify() {
  if (not isScalarType(getExpressionType(getCondition())))
    return emitOpError() << getOperationName()
                         << " condition requires a scalar type.";

  return mlir::success();
}

//===--------------------------- LocalVariableOp --------------------------===//

mlir::Value
LocalVariableOp::getBlockArgumentVariable(mlir::BlockArgument Argument) {
  return *this;
}

mlir::ParseResult LocalVariableOp::parse(mlir::OpAsmParser &Parser,
                                         mlir::OperationState &Result) {
  if (Parser.parseColon())
    return mlir::failure();

  mlir::Type Type;
  if (Parser.parseType(Type))
    return mlir::failure();

  auto Initializer = std::make_unique<::mlir::Region>();
  if (Parser.parseOptionalEqual().succeeded()) {
    llvm::SmallVector<mlir::OpAsmParser::Argument, 1> Arguments;

    if (Parser.parseOptionalLParen().succeeded()) {
      mlir::OpAsmParser::UnresolvedOperand Operand;
      if (Parser.parseOperand(Operand).failed())
        return mlir::failure();
      if (Parser.parseRParen().failed())
        return mlir::failure();
      Arguments.emplace_back(Operand, Type);
    }

    if (Parser.parseRegion(*Initializer, Arguments).failed())
      return mlir::failure();
  }

  if (Parser.parseOptionalAttrDictWithKeyword(Result.attributes))
    return mlir::failure();

  Result.addRegion(std::move(Initializer));
  Result.addTypes({ Type });

  return mlir::success();
}

void LocalVariableOp::print(mlir::OpAsmPrinter &Printer) {
  Printer << " : ";
  Printer << getType();

  if (mlir::Region &R = getInitializer(); not R.empty()) {
    Printer << " = ";
    if (R.getNumArguments() != 0) {
      Printer << '(';
      Printer.printOperand(R.getArgument(0));
      Printer << ')';
      Printer << ' ';
    }
    Printer.printRegion(R, /*printEntryBlockArgs=*/false);
  }

  Printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
}

mlir::LogicalResult LocalVariableOp::verify() {
  if (mlir::Region &R = getInitializer(); not R.empty()) {
    if (R.getNumArguments() != 0) {
      if (R.getNumArguments() != 1) {
        return emitOpError() << getOperationName()
                             << " initializer region may have no more than one"
                                " argument.";
      }

      if (R.getArgument(0).getType() != getType()) {
        return emitOpError() << getOperationName()
                             << " initializer region argument type must match "
                                " the local variable type.";
      }
    }

    if (getExpressionType(R) != removeConst(getType()))
      return emitOpError() << getOperationName()
                           << " initializer type must match the variable type";
  }

  return mlir::success();
}

//===----------------------------- MakeLabelOp ----------------------------===//

LabelAssignmentOpInterface MakeLabelOp::getAssignment() {
  for (mlir::OpOperand &Use : getResult().getUses()) {
    if (auto Op = mlir::dyn_cast<LabelAssignmentOpInterface>(Use.getOwner()))
      return Op;
  }
  return {};
}

static std::pair<size_t, size_t> getNumLabelUsers(MakeLabelOp Op) {
  size_t Assignments = 0;
  size_t Jumps = 0;
  for (mlir::OpOperand &Operand : Op.getResult().getUses()) {
    if (mlir::isa<LabelAssignmentOpInterface>(Operand.getOwner()))
      ++Assignments;
    else if (mlir::isa<JumpStatementOpInterface>(Operand.getOwner()))
      ++Jumps;
  }
  return { Assignments, Jumps };
}

mlir::LogicalResult MakeLabelOp::canonicalize(MakeLabelOp Op,
                                              mlir::PatternRewriter &Rewriter) {
  const auto [Assignments, Jumps] = getNumLabelUsers(Op);

  if (Jumps != 0)
    return mlir::failure();

  if (Assignments != 0) {
    mlir::Operation *AssignmentOp = Op.getAssignment();
    revng_assert(AssignmentOp != nullptr);

    if (auto AssignOp = mlir::dyn_cast<AssignLabelOp>(AssignmentOp)) {
      Rewriter.eraseOp(AssignOp);
    } else if (auto LoopOp = mlir::dyn_cast<LoopOpInterface>(AssignmentOp)) {
      if (Op.getResult() == LoopOp.getBreakLabel())
        LoopOp.setBreakLabel(nullptr);
      else if (Op.getResult() == LoopOp.getContinueLabel())
        LoopOp.setContinueLabel(nullptr);
    }
  }

  Rewriter.eraseOp(Op);
  return mlir::success();
}

mlir::LogicalResult MakeLabelOp::verify() {
  const auto [Assignments, Jumps] = getNumLabelUsers(*this);

  if (Assignments > 1)
    return emitOpError() << getOperationName()
                         << " may only have one assignment.";

  if (Jumps != 0 and Assignments == 0)
    return emitOpError() << getOperationName()
                         << " with a use by a jump operation must have an"
                            " assignment.";

  return mlir::success();
}

//===------------------------------ RequireOp -----------------------------===//

LocalVariableOp RequireOp::getLocalOp() {
  return getLocal().getDefiningOp<LocalVariableOp>();
}

mlir::LogicalResult RequireOp::verify() {
  if (not getLocalOp())
    return emitOpError() << getOperationName()
                         << " operand must be defined by clift.local.";

  return mlir::success();
}

//===------------------------------ ReturnOp ------------------------------===//

mlir::LogicalResult ReturnOp::verify() {
  mlir::Region &Expression = getResult();

  mlir::Type ExprType = Expression.empty() ? mlir::Type() :
                                             getExpressionType(Expression);

  if (ExprType and not clift::unwrapped_isa<VoidType, ValueType>(ExprType))
    return emitOpError() << getOperationName()
                         << " expression must have void or value type.";

  if (auto Function = getOperation()->getParentOfType<FunctionOp>()) {
    mlir::Type ReturnType = Function.getReturnType();
    mlir::Type UnqualifiedExprType = ExprType ? clift::removeConst(ExprType) :
                                                mlir::Type();

    if (clift::unwrapped_isa<VoidType>(ReturnType)) {
      if (ExprType)
        return emitOpError() << " cannot return expression in function "
                                " returning void.";
    } else if (UnqualifiedExprType != ReturnType) {
      return emitOpError() << getOperationName()
                           << " expression type must match the function return"
                              " type.";
    }
  }

  return mlir::success();
}

//===------------------------------ SwitchOp ------------------------------===//

bool SwitchOp::isIndirectlyNoFallthrough() const {
  return isIndirectlyNoFallthroughImpl(*this);
}

mlir::Type SwitchOp::getConditionType() {
  return getExpressionType(getConditionRegion());
}

void SwitchOp::build(mlir::OpBuilder &OdsBuilder,
                     mlir::OperationState &OdsState,
                     const llvm::ArrayRef<uint64_t> CaseValues) {
  llvm::SmallVector<int64_t> SignedCaseValues;
  SignedCaseValues.resize_for_overwrite(CaseValues.size());
  std::copy(CaseValues.begin(), CaseValues.end(), SignedCaseValues.begin());
  build(OdsBuilder, OdsState, SignedCaseValues, CaseValues.size());
}

mlir::ParseResult SwitchOp::parse(mlir::OpAsmParser &Parser,
                                  mlir::OperationState &Result) {
  // Condition region:
  Result.addRegion(std::make_unique<mlir::Region>());

  // Default case region:
  Result.addRegion(std::make_unique<mlir::Region>());

  if (Parser.parseRegion(*Result.regions[0]).failed())
    return Parser.emitError(Parser.getCurrentLocation(),
                            "Expected switch condition region");

  llvm::SmallVector<int64_t, 16> CaseValues;
  while (Parser.parseOptionalKeyword("case").succeeded()) {
    uint64_t CaseValue;
    if (Parser.parseInteger(CaseValue).failed())
      return Parser.emitError(Parser.getCurrentLocation(),
                              "Expected switch case value");

    auto R = std::make_unique<mlir::Region>();
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
                        mlir::DenseI64ArrayAttr::get(Parser.getContext(),
                                                     CaseValues));

  if (Parser.parseOptionalAttrDictWithKeyword(Result.attributes).failed())
    return mlir::failure();

  return mlir::success();
}

void SwitchOp::print(mlir::OpAsmPrinter &Printer) {
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

  Printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), Elided);
}

mlir::LogicalResult SwitchOp::verify() {
  if (not unwrapped_isa<IntegralType>(getExpressionType(getCondition())))
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

void WhileOp::build(mlir::OpBuilder &Builder,
                    mlir::OperationState &State,
                    LoopOpInterface OtherLoop) {
  buildLoop(Builder, State, 2, OtherLoop);
}

mlir::LogicalResult WhileOp::verify() {
  if (not isScalarType(getExpressionType(getCondition())))
    return emitOpError() << getOperationName()
                         << " condition requires a scalar type.";

  return mlir::success();
}

//===----------------------------- Expressions ----------------------------===//

//===------------------------------- YieldOp ------------------------------===//

bool YieldOp::isDiscardedOperand(mlir::OpOperand &Operand) {
  mlir::Region *R = getOperation()->getParentRegion();
  revng_assert(R != nullptr);

  auto Statement = mlir::cast<StatementOpInterface>(R->getParentOp());
  return Statement.isDiscardedExpression(*R);
}

bool YieldOp::isBooleanTestedOperand(mlir::OpOperand &Operand) {
  mlir::Region *R = getOperation()->getParentRegion();
  revng_assert(R != nullptr);

  auto Statement = mlir::cast<StatementOpInterface>(R->getParentOp());
  return Statement.isBooleanTestedExpression(*R);
}

//===------------------------------ StringOp ------------------------------===//

mlir::LogicalResult StringOp::verify() {
  auto ArrayT = mlir::dyn_cast<ArrayType>(getResult().getType());
  if (not ArrayT or not isConst(ArrayT))
    return emitOpError() << getOperationName()
                         << " result must have const array type.";

  auto CharT = mlir::dyn_cast<IntegerType>(ArrayT.getElementType());
  if (not CharT or CharT.getKind() != IntegerKind::Number
      or CharT.getSize() != 1)
    return emitOpError() << getOperationName()
                         << " result must have number8_t element type.";

  if (ArrayT.getElementsCount() != getValue().size() + 1)
    return emitOpError() << getOperationName()
                         << " result type length must match string length"
                            " (including null terminator).";

  return mlir::success();
}

//===------------------- Pointer arithmetic expressions -------------------===//

mlir::ParseResult static parseCliftPointerArithmeticOpTypes(mlir::OpAsmParser
                                                              &Parser,
                                                            mlir::Type &Result,
                                                            mlir::Type &Lhs,
                                                            mlir::Type &Rhs) {
  mlir::SMLoc TypesLoc = Parser.getCurrentLocation();

  if (Parser.parseType(Lhs).failed())
    return mlir::failure();

  if (Parser.parseComma().failed())
    return mlir::failure();

  if (Parser.parseType(Rhs).failed())
    return mlir::failure();

  auto LhsPT = clift::unwrapped_dyn_cast<PointerType>(Lhs);
  auto RhsPT = clift::unwrapped_dyn_cast<PointerType>(Rhs);

  if (static_cast<bool>(LhsPT) == static_cast<bool>(RhsPT))
    return Parser.emitError(TypesLoc, "Expected exactly one pointer type.");

  Result = clift::removeConst(LhsPT ? Lhs : Rhs);

  return mlir::success();
}

static void printCliftPointerArithmeticOpTypes(mlir::OpAsmPrinter &Printer,
                                               mlir::Operation *Op,
                                               mlir::Type Result,
                                               mlir::Type Lhs,
                                               mlir::Type Rhs) {
  Printer << Lhs;
  Printer << ',';
  Printer << Rhs;
}

static mlir::LogicalResult verifyPointerArithmeticOp(mlir::Operation *Op) {
  mlir::Type LhsT = Op->getOperand(0).getType();
  mlir::Type RhsT = Op->getOperand(1).getType();

  auto LhsPT = clift::unwrapped_dyn_cast<PointerType>(LhsT);
  auto RhsPT = clift::unwrapped_dyn_cast<PointerType>(RhsT);

  if (static_cast<bool>(LhsPT) == static_cast<bool>(RhsPT))
    return Op->emitOpError() << "requires exactly one pointer operand.";

  auto PtrType = LhsPT ? LhsPT : RhsPT;
  auto IntType = clift::unwrapped_dyn_cast<IntegerType>(LhsPT ? RhsT : LhsT);

  if (not IntType)
    return Op->emitOpError() << "requires an integer operand.";

  if (mlir::isa<PtrSubOp>(Op)) {
    if (not LhsPT)
      return Op->emitOpError() << "left operand must have pointer type.";
  }

  if (IntType.getSize() != PtrType.getPointerSize())
    return Op->emitOpError() << "pointer and integer operand sizes must "
                                "match.";

  if (not clift::unwrapped_isa<ObjectType>(PtrType.getPointeeType()))
    return Op->emitOpError() << "operand pointee must have object type.";

  if (Op->getResult(0).getType() != removeConst(LhsPT ? LhsT : RhsT))
    return Op->emitOpError() << "result and pointer operand types must match.";

  return mlir::success();
}

unsigned
clift::impl::getPointerArithmeticPointerOperandIndex(mlir::Operation *Op) {
  return clift::unwrapped_isa<PointerType>(Op->getOperand(0).getType()) ? 0 : 1;
}

unsigned
clift::impl::getPointerArithmeticOffsetOperandIndex(mlir::Operation *Op) {
  return clift::unwrapped_isa<PointerType>(Op->getOperand(0).getType()) ? 1 : 0;
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
  auto LHS = clift::unwrapped_cast<PointerType>(getLhs().getType());
  auto RHS = clift::unwrapped_cast<PointerType>(getRhs().getType());

  if (not clift::equivalent(clift::unwrapTypedefs(LHS.getPointeeType()),
                            clift::unwrapTypedefs(RHS.getPointeeType())))
    return emitOpError() << getOperationName()
                         << " operand pointee types must be equal (ignoring"
                            " cv-qualifiers).";

  if (not clift::unwrapped_isa<ObjectType>(LHS.getPointeeType()))
    return emitOpError() << getOperationName()
                         << " operand pointee must have object type.";

  auto IntType = mlir::dyn_cast<IntegerType>(getResult().getType());
  if (not IntType or not IntType.isSigned()
      or IntType.getSize() != LHS.getPointerSize())
    return emitOpError() << getOperationName()
                         << " result must have primitive signed integer type"
                            " with size matching that of the operand type.";

  return mlir::success();
}

//===------------------------------- DecayOp ------------------------------===//

mlir::LogicalResult DecayOp::verify() {
  auto ArgT = collapseTypedefs(getValue().getType());

  auto PtrT = clift::unwrapped_dyn_cast<PointerType>(getResult().getType());
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

  return mlir::success();
}

//===------------------------------ ExtendOp ------------------------------===//

mlir::LogicalResult ExtendOp::verify() {
  mlir::Type ResT = getResult().getType();
  mlir::Type ArgT = getValue().getType();

  if (getObjectSize(ArgT) >= getObjectSize(ResT))
    return emitOpError() << getOperationName()
                         << " result must be wider than the operand";

  return mlir::success();
}

//===----------------------------- TruncateOp -----------------------------===//

mlir::LogicalResult TruncateOp::verify() {
  mlir::Type ResT = getResult().getType();
  mlir::Type ArgT = getValue().getType();

  if (getObjectSize(ArgT) <= getObjectSize(ResT))
    return emitOpError() << getOperationName()
                         << " result must be narrower than the operand";

  return mlir::success();
}

//===----------------------------- PtrResizeOp ----------------------------===//

mlir::LogicalResult PtrResizeOp::verify() {
  auto ResT = getResult().getType();
  auto ArgT = getValue().getType();

  if (getObjectSize(ResT) == getObjectSize(ArgT))
    return emitOpError() << getOperationName()
                         << " operand size must not match the result.";

  return mlir::success();
}

//===------------------------------ AccessOp ------------------------------===//

bool AccessOp::isLvalueExpression() {
  return isIndirect() or clift::isLvalueExpression(getValue());
}

ClassType AccessOp::getClassType() {
  auto ObjectT = unwrapTypedefs(getValue().getType());

  if (isIndirect()) {
    ObjectT = mlir::cast<PointerType>(ObjectT).getPointeeType();
    ObjectT = unwrapTypedefs(ObjectT);
  }

  return mlir::cast<ClassType>(removeConst(ObjectT));
}

FieldAttr AccessOp::getFieldAttr() {
  return getClassType().getFields()[getMemberIndex()];
}

mlir::LogicalResult AccessOp::verify() {
  auto ObjectT = collapseTypedefs(getValue().getType());

  if (auto PointerT = mlir::dyn_cast<PointerType>(ObjectT)) {
    if (not isIndirect())
      return emitOpError() << getOperationName()
                           << " operand must have pointer type.";

    ObjectT = unwrapTypedefs(PointerT.getPointeeType());
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
  if (not clift::unwrapped_isa<ObjectType>(PointeeT))
    return emitOpError() << getOperationName()
                         << " cannot dereference pointer to non-object type.";

  if (getResult().getType() != PointeeT)
    return emitOpError() << getOperationName()
                         << " result type must match the pointer type.";

  return mlir::success();
}

//===-------------------------------- UseOp -------------------------------===//

mlir::LogicalResult
UseOp::verifySymbolUses(mlir::SymbolTableCollection &SymbolTable) {
  auto Module = getOperation()->getParentOfType<mlir::ModuleOp>();
  mlir::Operation *Op = SymbolTable.lookupSymbolIn(Module, getSymbolNameAttr());

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

FunctionType CallOp::getFunctionType() {
  return getFunctionOrFunctionPointerFunctionType(getFunction().getType());
}

namespace {

using DefaultArgumentTypeProvider = llvm::function_ref<mlir::Type(unsigned)>;

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
  mlir::ParseResult parse(mlir::OpAsmParser &Parser, bool RequireTypes) {
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

  mlir::ParseResult resolveOperands(mlir::OpAsmParser &Parser,
                                    mlir::OperationState &Result) {
    return Parser.resolveOperands(Operands, Types, Location, Result.operands);
  }

  mlir::ParseResult
  resolveOperands(mlir::OpAsmParser &Parser,
                  mlir::OperationState &Result,
                  DefaultArgumentTypeProvider GetDefaultType) {
    for (auto [I, T] : llvm::enumerate(Types)) {
      if (not T) {
        if (mlir::Type DefaultType = GetDefaultType(I))
          T = removeConst(DefaultType);
      }
    }

    return resolveOperands(Parser, Result);
  }

private:
  mlir::SMLoc Location;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> Operands;
  llvm::SmallVector<mlir::Type> Types;
};

} // namespace

static void printArgumentList(mlir::OpAsmPrinter &Printer,
                              mlir::OperandRange Operands,
                              DefaultArgumentTypeProvider GetDefaultType) {
  Printer << '(';
  for (auto [I, V] : llvm::enumerate(Operands)) {
    if (I != 0)
      Printer << ", ";

    Printer << V;
    if (mlir::Type DefaultType = GetDefaultType(I))
      if (V.getType() != removeConst(DefaultType))
        Printer << " : " << V.getType();
  }
  Printer << ')';
}

static auto makeCallArgumentTypeAccessor(clift::FunctionType Function) {
  return [Function](unsigned I) -> mlir::Type {
    auto ParameterTypes = Function.getArgumentTypes();
    return I < ParameterTypes.size() ? ParameterTypes[I] : mlir::Type();
  };
}

mlir::ParseResult CallOp::parse(mlir::OpAsmParser &Parser,
                                mlir::OperationState &Result) {
  mlir::OpAsmParser::UnresolvedOperand FunctionOperand;
  if (Parser.parseOperand(FunctionOperand).failed())
    return mlir::failure();

  ArgumentListParser Arguments;
  if (Arguments.parse(Parser, /*RequireTypes=*/false).failed())
    return mlir::failure();

  if (Parser.parseOptionalAttrDict(Result.attributes).failed())
    return mlir::failure();

  if (Parser.parseColon().failed())
    return mlir::failure();

  mlir::SMLoc FuncTypeLoc = Parser.getCurrentLocation();
  mlir::Type FuncType;
  if (Parser.parseType(FuncType).failed())
    return mlir::failure();

  auto FunctionType = getFunctionOrFunctionPointerFunctionType(FuncType);

  if (not FunctionType)
    return Parser.emitError(FuncTypeLoc) << "expected Clift function or "
                                            "pointer-to-function type";

  Result.addTypes(FunctionType.getResultTypes());

  if (Parser.resolveOperand(FunctionOperand, FuncType, Result.operands)
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

void CallOp::print(mlir::OpAsmPrinter &Printer) {
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
  auto FuncType = getFunctionOrFunctionPointerFunctionType(getFunction()
                                                             .getType());
  if (not FuncType)
    return emitOpError() << getOperationName()
                         << " function argument must have function or pointer"
                         << "-to-function type.";

  auto ArgumentTypes = getArguments().getTypes();
  auto ParameterTypes = FuncType.getArgumentTypes();

  if (ArgumentTypes.size() != ParameterTypes.size())
    return emitOpError() << getOperationName()
                         << " argument count must match the number of function"
                            " parameters.";

  for (auto &&[ArgumentT, ParameterT] :
       llvm::zip_equal(ArgumentTypes, ParameterTypes)) {
    if (removeConst(ArgumentT) != removeConst(ParameterT))
      return emitOpError() << getOperationName()
                           << " argument types must match the parameter types"
                              " of the function, ignoring qualifiers.";
  }

  if (getResult().getType() != removeConst(FuncType.getReturnType()))
    return emitOpError() << getOperationName()
                         << " result type must match the return type of the"
                            " function, ignoring qualifiers.";

  return mlir::success();
}

//===------------------------------ TernaryOp -----------------------------===//

static mlir::ParseResult parseCliftTernaryOpTypes(mlir::OpAsmParser &Parser,
                                                  mlir::Type &Condition,
                                                  mlir::Type &Lhs,
                                                  mlir::Type &Rhs) {
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

static void printCliftTernaryOpTypes(mlir::OpAsmPrinter &Printer,
                                     mlir::Operation *Op,
                                     mlir::Type Condition,
                                     mlir::Type Lhs,
                                     mlir::Type Rhs) {
  Printer << Condition;
  Printer << ',';
  Printer << Lhs;

  if (Lhs != Rhs) {
    Printer << ',';
    Printer << Rhs;
  }
}

//===----------------------------- AggregateOp ----------------------------===//

static auto makeAggregateArgumentTypeAccessor(mlir::Type Type) {
  auto UnderlyingType = unwrapTypedefs(Type);
  return [UnderlyingType](unsigned I) -> mlir::Type {
    if (auto Array = mlir::dyn_cast<ArrayType>(UnderlyingType))
      return Array.getElementType();

    if (auto Struct = mlir::dyn_cast<StructType>(UnderlyingType)) {
      auto Fields = Struct.getFields();
      return I < Fields.size() ? Fields[I].getType() : mlir::Type();
    }

    return {};
  };
}

mlir::ParseResult AggregateOp::parse(mlir::OpAsmParser &Parser,
                                     mlir::OperationState &Result) {
  ArgumentListParser Arguments;
  if (Arguments.parse(Parser, /*Requiretypes=*/false).failed())
    return mlir::failure();

  if (Parser.parseOptionalAttrDict(Result.attributes).failed())
    return mlir::failure();

  if (Parser.parseColon().failed())
    return mlir::failure();

  mlir::Type ResultType;
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

void AggregateOp::print(mlir::OpAsmPrinter &Printer) {
  mlir::Type ResultType = getResult().getType();

  printArgumentList(Printer,
                    getInitializers(),
                    makeAggregateArgumentTypeAccessor(ResultType));

  Printer.printOptionalAttrDict(getOperation()->getAttrs(), {});

  Printer << " : ";
  Printer << ResultType;
}

mlir::LogicalResult AggregateOp::verify() {
  auto InitializerTypes = getInitializers().getTypes();
  auto AT = unwrapTypedefs(getResult().getType());

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
