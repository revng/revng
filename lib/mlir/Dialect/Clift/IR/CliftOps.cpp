//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallSet.h"

#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/RegionGraphTraits.h"

#include "revng/Support/GraphAlgorithms.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"

#define GET_OP_CLASSES
#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.cpp.inc"

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

static FunctionTypeAttr getFunctionTypeAttr(mlir::Type Type) {
  auto T = mlir::cast<DefinedType>(dealias(Type));
  return mlir::cast<FunctionTypeAttr>(T.getElementType());
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

struct ModuleValidator {
  explicit ModuleValidator(clift::ModuleOp Module) : Module(Module) {}

  enum class LoopOrSwitch : uint8_t {
    Loop,
    Switch,
  };

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

  mlir::LogicalResult visitNestedOp(mlir::Operation *Op) {
    if (Op == ModuleLevelOp)
      return mlir::success();

    if (isModuleLevelOperation(Op))
      return Op->emitOpError() << Op->getName()
                               << " must be directly nested within a "
                                  "ModuleOp.";

    if (auto Return = mlir::dyn_cast<ReturnOp>(Op)) {
      ValueType ReturnType = {};

      if (Region &R = Return.getResult(); not R.empty())
        ReturnType = getExpressionType(R);

      if (ReturnType and isVoid(FunctionReturnType))
        return Op->emitOpError() << Op->getName()
                                 << " cannot return expression in function "
                                    "returning void.";

      if (ReturnType != FunctionReturnType)
        return Op->emitOpError() << Op->getName()
                                 << " type does not match the function return "
                                    "type";
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
    }

    return visitOp(Op);
  }

  mlir::LogicalResult visitModuleLevelOp(mlir::Operation *Op) {
    ModuleLevelOp = Op;

    if (not isModuleLevelOperation(Op)) {
      return Op->emitOpError() << Op->getName()
                               << " cannot be directly nested within a "
                                  "ModuleOp.";
    }

    if (auto F = mlir::dyn_cast<FunctionOp>(Op)) {
      auto TypeAttr = getFunctionTypeAttr(F.getFunctionType());
      FunctionReturnType = mlir::cast<ValueType>(TypeAttr.getReturnType());
    }

    return visitOp(Op);
  }

private:
  clift::ModuleOp Module;
  Operation *ModuleLevelOp = nullptr;
  clift::ValueType FunctionReturnType;
  llvm::SmallPtrSet<mlir::Type, 32> VisitedTypes;
  llvm::SmallPtrSet<mlir::Attribute, 32> VisitedAttrs;
  llvm::DenseMap<uint64_t, TypeDefinitionAttr> Definitions;

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
    while (Op != Module.getOperation()) {
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
  ModuleValidator Validator(*this);

  Region &R = getRegion();

  if (not R.hasOneBlock())
    return emitOpError() << getOperationName()
                         << " must contain exactly one block.";

  for (Operation &Op : R.front()) {
    if (mlir::failed(Validator.visitModuleLevelOp(&Op)))
      return mlir::failure();

    const auto Visitor = [&](Operation *NestedOp) -> mlir::WalkResult {
      return Validator.visitNestedOp(NestedOp);
    };

    if (Op.walk(Visitor).wasInterrupted())
      return mlir::failure();
  }

  if (mlir::failed(Validator.visitOp(getOperation())))
    return mlir::failure();

  return mlir::success();
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
  if (not isReturnableType(getExpressionType(getResult())))
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
                         << " must yield an object type or void.";

  return mlir::success();
}
