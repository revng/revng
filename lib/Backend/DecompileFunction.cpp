//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/EarlyFunctionAnalysis/FunctionMetadataCache.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/Identifier.h"
#include "revng/Model/PrimitiveTypeKind.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Model/Segment.h"
#include "revng/Model/StructType.h"
#include "revng/Model/Type.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/IndentedOstream.h"
#include "revng/PTML/ModelHelpers.h"
#include "revng/Pipeline/Location.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/YAMLTraits.h"

#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/InitModelTypes/InitModelTypes.h"
#include "revng-c/Pipes/Ranks.h"
#include "revng-c/RestructureCFG/ASTNode.h"
#include "revng-c/RestructureCFG/ASTTree.h"
#include "revng-c/RestructureCFG/BeautifyGHAST.h"
#include "revng-c/RestructureCFG/RestructureCFG.h"
#include "revng-c/Support/DecompilationHelpers.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/Support/PTMLC.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"
#include "revng-c/TypeNames/ModelToPTMLTypeHelpers.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

using llvm::BasicBlock;
using llvm::CallInst;
using llvm::Instruction;
using llvm::raw_ostream;
using llvm::StringRef;

using model::Binary;
using model::CABIFunctionType;
using model::QualifiedType;
using model::Qualifier;
using model::RawFunctionType;
using model::TypedefType;

using modelEditPath::getCustomNamePath;
using pipeline::serializedLocation;
using ptml::str;
using ptml::Tag;
namespace ranks = revng::ranks;
namespace attributes = ptml::attributes;
namespace tokens = ptml::c::tokens;
namespace tags = ptml::tags;

using tokenDefinition::types::StringToken;
using tokenDefinition::types::TypeString;
using TokenMapT = std::map<const llvm::Value *, std::string>;
using ModelTypesMap = std::map<const llvm::Value *, const model::QualifiedType>;
using InstrSetVec = llvm::SmallSetVector<const llvm::Instruction *, 8>;
using InlineableTypesMap = std::unordered_map<const model::Function *,
                                              std::set<const model::Type *>>;
using QualifiedTypeNameMap = std::map<model::QualifiedType, std::string>;
using TypeToNumOfRefsMap = std::unordered_map<const model::Type *, unsigned>;
using GraphInfo = TypeInlineHelper::GraphInfo;

static constexpr const char *StackFrameVarName = "stack";

static Logger<> Log{ "c-backend" };
static Logger<> VisitLog{ "c-backend-visit-order" };

static bool isAssignment(const llvm::Value *I) {
  return isCallToTagged(I, FunctionTags::Assign);
}

static bool isLocalVarDecl(const llvm::Value *I) {
  return isCallToTagged(I, FunctionTags::LocalVariable);
}

static bool isCallStackArgumentDecl(const llvm::Value *I) {
  auto *Call = dyn_cast_or_null<llvm::CallInst>(I);
  if (not Call)
    return false;

  auto *Callee = Call->getCalledFunction();
  if (not Callee)
    return false;

  return Callee->getName().startswith("revng_call_stack_arguments");
}

static bool isStackFrameDecl(const llvm::Value *I) {
  auto *Call = dyn_cast_or_null<llvm::CallInst>(I);
  if (not Call)
    return false;

  auto *Callee = Call->getCalledFunction();
  if (not Callee)
    return false;

  return Callee->getName().startswith("revng_stack_frame");
}

static const llvm::CallInst *isCallToNonIsolated(const llvm::Instruction *I) {
  if (isCallToTagged(I, FunctionTags::QEMU)
      or isCallToTagged(I, FunctionTags::Helper)
      or isCallToTagged(I, FunctionTags::Exceptional)
      or llvm::isa<llvm::IntrinsicInst>(I))
    return llvm::cast<CallInst>(I);

  return nullptr;
}

static bool isCallToCustomOpcode(const llvm::Instruction *I) {
  return isCallToTagged(I, FunctionTags::Copy)
         or isCallToTagged(I, FunctionTags::Assign)
         or isCallToTagged(I, FunctionTags::ModelCast)
         or isCallToTagged(I, FunctionTags::ModelGEP)
         or isCallToTagged(I, FunctionTags::ModelGEPRef)
         or isCallToTagged(I, FunctionTags::AddressOf)
         or isCallToTagged(I, FunctionTags::Parentheses)
         or isCallToTagged(I, FunctionTags::OpaqueCSVValue)
         or isCallToTagged(I, FunctionTags::StructInitializer)
         or isCallToTagged(I, FunctionTags::SegmentRef)
         or isCallToTagged(I, FunctionTags::UnaryMinus)
         or isCallToTagged(I, FunctionTags::BinaryNot)
         or isCallToTagged(I, FunctionTags::BooleanNot)
         or isCallToTagged(I, FunctionTags::StringLiteral);
}

static bool isIntegerConstFormatting(const llvm::Value *Call) {
  return isCallToTagged(Call, FunctionTags::HexInteger)
         or isCallToTagged(Call, FunctionTags::CharInteger)
         or isCallToTagged(Call, FunctionTags::BoolInteger);
}

static bool isCConstant(const llvm::Value *V) {
  return isa<llvm::Constant>(V) or isIntegerConstFormatting(V);
}

static std::string addAlwaysParentheses(llvm::StringRef Expr) {
  return std::string("(") + Expr.str() + ")";
}

static std::string
get128BitIntegerHexConstant(llvm::APInt Value,
                            const ptml::PTMLCBuilder &ThePTMLCBuilder) {
  revng_assert(Value.getBitWidth() > 64);
  // In C, even if you can have 128-bit variables, you cannot have 128-bit
  // literals, so we need this hack to assign a big constant value to a
  // 128-bit variable.
  llvm::APInt LowBits = Value.getLoBits(64);
  llvm::APInt HighBits = Value.getHiBits(Value.getBitWidth() - 64);

  StringToken LowBitsString;
  LowBits.toString(LowBitsString,
                   /*radix=*/16,
                   /*signed=*/false,
                   /*formatAsCLiteral=*/true);
  StringToken HighBitsString;
  HighBits.toString(HighBitsString,
                    /*radix=*/16,
                    /*signed=*/false,
                    /*formatAsCLiteral=*/true);

  using PTMLOperator = ptml::PTMLCBuilder::Operator;
  auto HighConstant = ThePTMLCBuilder.getConstantTag(HighBitsString) + " "
                      + ThePTMLCBuilder.getOperator(PTMLOperator::LShift) + " "
                      + ThePTMLCBuilder.getNumber(64);
  auto CompositeConstant = HighConstant + " "
                           + ThePTMLCBuilder.getOperator(PTMLOperator::Or) + " "
                           + ThePTMLCBuilder.getConstantTag(LowBitsString);
  return addAlwaysParentheses(CompositeConstant);
}

static std::string hexLiteral(const llvm::ConstantInt *Int,
                              const ptml::PTMLCBuilder &ThePTMLCBuilder) {
  StringToken Formatted;
  if (Int->getBitWidth() <= 64) {
    Int->getValue().toString(Formatted,
                             /*radix*/ 16,
                             /*signed*/ false,
                             /*formatAsCLiteral*/ true);
    return Formatted.str().str();
  }
  return get128BitIntegerHexConstant(Int->getValue(), ThePTMLCBuilder);
}

static std::string charLiteral(const llvm::ConstantInt *Int) {
  revng_assert(Int->getValue().getBitWidth() == 8);
  const auto LimitedValue = Int->getLimitedValue(0xffu);
  const auto CharValue = static_cast<char>(LimitedValue);

  std::string EscapedC;
  llvm::raw_string_ostream EscapeCStream(EscapedC);
  EscapeCStream.write_escaped(std::string(&CharValue, 1));

  std::string EscapedHTML;
  llvm::raw_string_ostream EscapeHTMLStream(EscapedHTML);
  llvm::printHTMLEscaped(EscapedC, EscapeHTMLStream);

  return llvm::formatv("'{0}'", EscapeHTMLStream.str());
}

static std::string boolLiteral(const llvm::ConstantInt *Int) {
  revng_assert(Int->getBitWidth() == 1);
  if (Int->isZero()) {
    return "false";
  } else {
    return "true";
  }
}

struct CCodeGenerator {
private:
  /// The model of the binary being analysed
  const Binary &Model;
  /// The LLVM function that is being decompiled
  const llvm::Function &LLVMFunction;
  /// The model function corresponding to LLVMFunction
  const model::Function &ModelFunction;
  /// The model prototype of ModelFunction
  const model::Type &ParentPrototype;
  /// The (combed) control flow AST
  const ASTTree &GHAST;

  /// Set of values that have a corresponding local variable which should be
  /// declared at the start of the function
  const InstrSetVec &TopScopeVariables;
  /// A map containing a model type for each LLVM value in the function
  const ModelTypesMap TypeMap;

  /// Where to output the decompiled C code
  ptml::PTMLIndentedOstream Out;

  ptml::PTMLCBuilder ThePTMLCBuilder;

  /// Name of the local variable used to break out of loops from within nested
  /// switches
  std::vector<std::string> SwitchStateVars;

  FunctionMetadataCache &Cache;

private:
  class VarNameGenerator {
  private:
    uint64_t CurVarID = 0;

  public:
    std::string nextVarName() { return "var_" + to_string(CurVarID++); }

    StringToken nextSwitchStateVar() {
      StringToken StateVar("break_from_loop_");
      StateVar += to_string(CurVarID++);
      return StateVar;
    }
  };

  /// Stateful generator for variable names
  VarNameGenerator NameGenerator;

  /// Keep track of the names associated with function arguments, and local
  /// variables. In the past it also kept track of intermediate expressions, but
  /// with the new design all the tokens corresponding to instructions that
  /// don't represent local variables are recomputed every time.
  TokenMapT TokenMap;

private:
  /// Name of the local variable used to break out from loops
  std::string LoopStateVar;
  std::string LoopStateVarDeclaration;

private:
  /// Emission of parentheses may change whether the OPRP is enabled or not
  bool IsOperatorPrecedenceResolutionPassEnabled = false;

public:
  CCodeGenerator(FunctionMetadataCache &Cache,
                 const Binary &Model,
                 const llvm::Function &LLVMFunction,
                 const ASTTree &GHAST,
                 const InstrSetVec &TopScopeVariables,
                 raw_ostream &Out,
                 ptml::PTMLCBuilder &ThePTMLCBuilder) :
    Model(Model),
    LLVMFunction(LLVMFunction),
    ModelFunction(*llvmToModelFunction(Model, LLVMFunction)),
    ParentPrototype(*ModelFunction.Prototype().getConst()),
    GHAST(GHAST),
    TopScopeVariables(TopScopeVariables),
    TypeMap(initModelTypes(Cache,
                           LLVMFunction,
                           &ModelFunction,
                           Model,
                           /*PointersOnly=*/false)),
    Out(Out, 4),
    ThePTMLCBuilder(ThePTMLCBuilder),
    SwitchStateVars(),
    Cache(Cache) {
    // TODO: don't use a global loop state variable
    LoopStateVar = getVariableLocationReference("loop_state_var",
                                                ModelFunction,
                                                ThePTMLCBuilder);
    LoopStateVarDeclaration = getVariableLocationDefinition("loop_state_var",
                                                            ModelFunction,
                                                            ThePTMLCBuilder);

    if (LLVMFunction.getMetadata(ExplicitParenthesesMDName))
      IsOperatorPrecedenceResolutionPassEnabled = true;
  }

  void emitFunction(bool NeedsLocalStateVar, InlineableTypesMap &StackTypes);

private:
  /// Visit a GHAST node and all its children recursively, emitting BBs
  /// and control flow statements in the process.
  RecursiveCoroutine<void> emitGHASTNode(const ASTNode *Node);

  /// Recursively build a C string representing the condition contained
  /// in an ExprNode (which might be composed by one or more subexpressions).
  /// Whenever an atomic node is encountered, the associated basic block is
  /// emitted on-the-fly.
  RecursiveCoroutine<std::string> buildGHASTCondition(const ExprNode *E);

  /// Serialize a basic block into a series of C statements.
  void emitBasicBlock(const BasicBlock *BB);

private:
  RecursiveCoroutine<std::string> getToken(const llvm::Value *V) const;

  RecursiveCoroutine<std::string>
  getCallToken(const llvm::CallInst *Call,
               const llvm::StringRef FuncName,
               const model::Type *Prototype) const;

  RecursiveCoroutine<std::string> getConstantToken(const llvm::Value *V) const;

  RecursiveCoroutine<std::string>
  getInstructionToken(const llvm::Instruction *I) const;

  RecursiveCoroutine<std::string>
  getCustomOpcodeToken(const llvm::CallInst *C) const;

  RecursiveCoroutine<std::string>
  getModelGEPToken(const llvm::CallInst *C) const;

  RecursiveCoroutine<std::string>
  getIsolatedCallToken(const llvm::CallInst *C) const;

  RecursiveCoroutine<std::string>
  getNonIsolatedCallToken(const llvm::CallInst *C) const;

private:
  std::string addParentheses(llvm::StringRef Expr) const;

  std::string buildDerefExpr(llvm::StringRef Expr) const;

  std::string buildAddressExpr(llvm::StringRef Expr) const;

  /// Return a C string that represents a cast of \a ExprToCast to a given
  /// \a DestType. If no casting is needed between the two expression, the
  /// original expression is returned.
  std::string buildCastExpr(StringRef ExprToCast,
                            const model::QualifiedType &SrcType,
                            const model::QualifiedType &DestType) const;

private:
  std::string createTopScopeVarDeclName(const llvm::Instruction *I) {
    revng_assert(isStackFrameDecl(I) or TopScopeVariables.contains(I));
    revng_assert(not TokenMap.contains(I));

    std::string VarName = isStackFrameDecl(I) ? std::string(StackFrameVarName) :
                                                NameGenerator.nextVarName();

    TokenMap[I] = getVariableLocationReference(VarName,
                                               ModelFunction,
                                               ThePTMLCBuilder);
    return getVariableLocationDefinition(VarName,
                                         ModelFunction,
                                         ThePTMLCBuilder);
  }

  std::string createLocalVarDeclName(const llvm::Instruction *I) {
    revng_assert(isLocalVarDecl(I) or isCallStackArgumentDecl(I));
    std::string VarName = NameGenerator.nextVarName();
    // This may override the entry for I, if I belongs to a "duplicated"
    // BasicBlock that is reachable from many paths on the GHAST.
    TokenMap[I] = getVariableLocationReference(VarName,
                                               ModelFunction,
                                               ThePTMLCBuilder);
    return getVariableLocationDefinition(VarName,
                                         ModelFunction,
                                         ThePTMLCBuilder);
  }

  std::string getVarName(const llvm::Instruction *I) const {
    revng_assert(isStackFrameDecl(I) or isLocalVarDecl(I)
                 or isCallStackArgumentDecl(I));
    revng_assert(TokenMap.contains(I));
    return TokenMap.at(I);
  };
};

std::string CCodeGenerator::addParentheses(llvm::StringRef Expr) const {
  if (IsOperatorPrecedenceResolutionPassEnabled)
    return Expr.str();
  return addAlwaysParentheses(Expr);
}

std::string CCodeGenerator::buildDerefExpr(llvm::StringRef Expr) const {
  using PTMLOperator = ptml::PTMLCBuilder::Operator;
  return ThePTMLCBuilder.getOperator(PTMLOperator::PointerDereference)
         + addParentheses(Expr);
}

std::string CCodeGenerator::buildAddressExpr(llvm::StringRef Expr) const {
  return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::AddressOf)
         + addParentheses(Expr);
}

std::string
CCodeGenerator::buildCastExpr(StringRef ExprToCast,
                              const model::QualifiedType &SrcType,
                              const model::QualifiedType &DestType) const {
  if (SrcType == DestType or not SrcType.UnqualifiedType().isValid()
      or not DestType.UnqualifiedType().isValid())
    return ExprToCast.str();

  revng_assert((SrcType.isScalar() or SrcType.isPointer())
               and (DestType.isScalar() or DestType.isPointer()));

  return addAlwaysParentheses(getTypeName(DestType, ThePTMLCBuilder)) + " "
         + addParentheses(ExprToCast);
}

static std::string getInvalidToken(const llvm::UndefValue *U,
                                   const ptml::PTMLCBuilder &ThePTMLCBuilder) {
  revng_assert(U->getType()->isIntOrPtrTy());

  std::string Result = ThePTMLCBuilder.getZeroTag().serialize() + ' ';
  if (isa<llvm::PoisonValue>(U))
    Result += " " + ThePTMLCBuilder.getBlockComment("poison", false) + " ";
  else
    Result += " " + ThePTMLCBuilder.getBlockComment("undef", false) + " ";
  return Result;
}

static std::string
getFormattedIntegerToken(const llvm::CallInst *Call,
                         const ptml::PTMLCBuilder &ThePTMLCBuilder) {

  if (isCallToTagged(Call, FunctionTags::HexInteger)) {
    const auto Operand = Call->getArgOperand(0);
    const auto *Value = cast<llvm::ConstantInt>(Operand);
    return ThePTMLCBuilder.getConstantTag(hexLiteral(Value, ThePTMLCBuilder))
      .serialize();
  }

  if (isCallToTagged(Call, FunctionTags::CharInteger)) {
    const auto Operand = Call->getArgOperand(0);
    const auto *Value = cast<llvm::ConstantInt>(Operand);
    return ThePTMLCBuilder.getConstantTag(charLiteral(Value)).serialize();
  }

  if (isCallToTagged(Call, FunctionTags::BoolInteger)) {
    const auto Operand = Call->getArgOperand(0);
    const auto *Value = cast<llvm::ConstantInt>(Operand);
    return ThePTMLCBuilder.getConstantTag(boolLiteral(Value)).serialize();
  }

  std::string Error = "Cannot get token for custom opcode: "
                      + dumpToString(Call);
  revng_abort(Error.c_str());
  return "";
}

RecursiveCoroutine<std::string>
CCodeGenerator::getConstantToken(const llvm::Value *C) const {
  revng_assert(isCConstant(C));

  if (auto *Undef = dyn_cast<llvm::UndefValue>(C))
    rc_return getInvalidToken(Undef, ThePTMLCBuilder);

  if (auto *Null = dyn_cast<llvm::ConstantPointerNull>(C))
    rc_return ThePTMLCBuilder.getNullTag().serialize();

  if (auto *Const = dyn_cast<llvm::ConstantInt>(C)) {
    llvm::APInt Value = Const->getValue();
    if (Value.isIntN(64))
      rc_return ThePTMLCBuilder.getNumber(Value.getLimitedValue()).serialize();
    else
      rc_return get128BitIntegerHexConstant(Value, ThePTMLCBuilder);
  }

  if (auto *Global = dyn_cast<llvm::GlobalVariable>(C)) {
    using namespace llvm;
    // Check if initializer is a CString
    auto *Initializer = Global->getInitializer();

    StringRef Content = "";
    if (auto StringInit = dyn_cast<ConstantDataArray>(Initializer)) {

      // If it's not a C string, bail out
      if (not StringInit->isCString())
        revng_abort(dumpToString(Global).c_str());

      // If it's a C string, Drop the terminator
      Content = StringInit->getAsString().drop_back();
    } else {
      // Zero initializers are always valid c empty strings, in all the
      // other cases, bail out
      if (not isa<llvm::ConstantAggregateZero>(Initializer))
        revng_abort(dumpToString(Global).c_str());
    }

    std::string Escaped;
    {
      raw_string_ostream Stream(Escaped);
      Stream << "\"";
      Stream.write_escaped(Content);
      Stream << "\"";
    }

    rc_return Escaped;
  }

  if (auto *ConstExpr = dyn_cast<llvm::ConstantExpr>(C)) {
    switch (ConstExpr->getOpcode()) {

    case Instruction::IntToPtr: {
      const auto *Operand = cast<llvm::Constant>(ConstExpr->getOperand(0));
      const QualifiedType &SrcType = TypeMap.at(Operand);
      const QualifiedType &DstType = TypeMap.at(ConstExpr);

      // IntToPtr has no effect on values that we already know to be pointers
      if (SrcType.isPointer())
        rc_return rc_recur getConstantToken(Operand);
      else
        rc_return buildCastExpr(rc_recur getConstantToken(Operand),
                                SrcType,
                                DstType);
    } break;

    default:
      revng_abort(dumpToString(ConstExpr).c_str());
    }
  }

  if (isIntegerConstFormatting(C))
    rc_return getFormattedIntegerToken(cast<llvm::CallInst>(C),
                                       ThePTMLCBuilder);

  std::string Error = "Cannot get token for llvm::Constant: ";
  Error += dumpToString(C).c_str();
  revng_abort(Error.c_str());

  rc_return "";
}

/// Traverse all nested typedefs inside \a QT, skipping const Qualifiers, and
/// returns a QualifiedType that represents the full traversal.
static RecursiveCoroutine<QualifiedType>
flattenTypedefsIgnoringConst(const QualifiedType &QT) {
  QualifiedType Result = peelConstAndTypedefs(QT);
  if (auto *TD = dyn_cast<TypedefType>(QT.UnqualifiedType().getConst())) {
    auto &Underlying = TD->UnderlyingType();
    QualifiedType Nested = rc_recur flattenTypedefsIgnoringConst(Underlying);
    Result.UnqualifiedType() = Nested.UnqualifiedType();
    llvm::move(Nested.Qualifiers(), std::back_inserter(Result.Qualifiers()));
  }
  rc_return Result;
}

RecursiveCoroutine<std::string>
CCodeGenerator::getModelGEPToken(const llvm::CallInst *Call) const {

  revng_assert(isCallToTagged(Call, FunctionTags::ModelGEP)
               or isCallToTagged(Call, FunctionTags::ModelGEPRef));

  revng_assert(Call->arg_size() >= 2);

  bool IsRef = isCallToTagged(Call, FunctionTags::ModelGEPRef);

  // First argument is a string containing the base type
  auto *CurArg = Call->arg_begin();
  QualifiedType CurType = deserializeFromLLVMString(CurArg->get(), Model);

  // Second argument is the base llvm::Value
  ++CurArg;
  llvm::Value *BaseValue = CurArg->get();
  std::string BaseString = rc_recur getToken(BaseValue);

  if (IsRef) {
    // In ModelGEPRefs, the base value is a reference, and the base type is
    // its type
    revng_assert(TypeMap.at(BaseValue) == CurType,
                 "The ModelGEP base type is not coherent with the "
                 "propagated type.");
  } else {
    // In ModelGEPs, the base value is a pointer, and the base type is the
    // type pointed by the base value
    QualifiedType PointerQt = CurType.getPointerTo(Model.Architecture());
    revng_assert(TypeMap.at(BaseValue) == PointerQt,
                 "The ModelGEP base type is not coherent with the "
                 "propagated type.");
  }

  // Arguments from the third represent indices for accessing
  // struct/unions/arrays
  ++CurArg;
  if (CurArg == Call->arg_end()) {
    if (not IsRef) {
      // If there are no further arguments, we are just dereferencing the
      // base value
      rc_return buildDerefExpr(BaseString);
    } else {
      // Dereferencing a reference does not produce any code
      rc_return BaseString;
    }
  }

  std::string CurExpr = addParentheses(BaseString);
  using PTMLOperator = ptml::PTMLCBuilder::Operator;
  Tag DerefSymbol = IsRef ? ThePTMLCBuilder.getOperator(PTMLOperator::Dot) :
                            ThePTMLCBuilder.getOperator(PTMLOperator::Arrow);

  // Traverse the model to decide whether to emit "." or "[]"
  for (; CurArg != Call->arg_end(); ++CurArg) {
    CurType = flattenTypedefsIgnoringConst(CurType);

    model::Qualifier *MainQualifier = nullptr;
    if (CurType.Qualifiers().size() > 0)
      MainQualifier = &CurType.Qualifiers().front();

    if (MainQualifier and model::Qualifier::isArray(*MainQualifier)) {
      // If it's an array, add "[]"

      std::string IndexExpr;
      if (auto *Const = dyn_cast<llvm::ConstantInt>(CurArg->get())) {
        IndexExpr = ThePTMLCBuilder.getNumber(Const->getValue()).serialize();
      } else {
        IndexExpr = rc_recur getToken(CurArg->get());
      }

      CurExpr += "[" + IndexExpr + "]";
      // Remove the qualifier we just analysed
      auto RemainingQualifiers = llvm::drop_begin(CurType.Qualifiers(), 1);
      CurType.Qualifiers() = { RemainingQualifiers.begin(),
                               RemainingQualifiers.end() };
    } else {
      // We shouldn't be going past pointers in a single ModelGEP
      revng_assert(not MainQualifier);

      // If it's a struct or union, we can only navigate it with fixed
      // indexes.
      // TODO: decide how to emit constants
      auto *FieldIdxConst = cast<llvm::ConstantInt>(CurArg->get());
      uint64_t FieldIdx = FieldIdxConst->getValue().getLimitedValue();

      CurExpr += DerefSymbol.serialize();

      // Find the field name
      const auto *UnqualType = CurType.UnqualifiedType().getConst();

      if (auto *Struct = dyn_cast<model::StructType>(UnqualType)) {
        const model::StructField &Field = Struct->Fields().at(FieldIdx);
        CurExpr += ThePTMLCBuilder.getLocationReference(*Struct, Field);
        CurType = Struct->Fields().at(FieldIdx).Type();

      } else if (auto *Union = dyn_cast<model::UnionType>(UnqualType)) {
        const model::UnionField &Field = Union->Fields().at(FieldIdx);
        CurExpr += ThePTMLCBuilder.getLocationReference(*Union, Field);
        CurType = Union->Fields().at(FieldIdx).Type();

      } else {
        revng_abort("Unexpected ModelGEP type found: ");
        CurType.dump();
      }
    }

    // Regardless if the base type was a pointer or not, we are now
    // navigating only references
    DerefSymbol = ThePTMLCBuilder.getOperator(PTMLOperator::Dot);
  }

  rc_return CurExpr;
}

RecursiveCoroutine<std::string>
CCodeGenerator::getCustomOpcodeToken(const llvm::CallInst *Call) const {

  if (isAssignment(Call)) {
    const llvm::Value *StoredVal = Call->getArgOperand(0);
    const llvm::Value *PointerVal = Call->getArgOperand(1);
    rc_return rc_recur getToken(PointerVal) + " "
      + ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::Assign) + " "
      + rc_recur getToken(StoredVal);
  }

  if (isCallToTagged(Call, FunctionTags::Copy))
    rc_return rc_recur getToken(Call->getArgOperand(0));

  if (isCallToTagged(Call, FunctionTags::ModelGEP)
      or isCallToTagged(Call, FunctionTags::ModelGEPRef))
    rc_return rc_recur getModelGEPToken(Call);

  if (isCallToTagged(Call, FunctionTags::ModelCast)) {
    // First argument is a string containing the base type
    auto *CurArg = Call->arg_begin();
    QualifiedType CurType = deserializeFromLLVMString(CurArg->get(), Model);

    // Second argument is the base llvm::Value
    ++CurArg;
    llvm::Value *BaseValue = CurArg->get();

    // Emit the parenthesized cast expr, and we are done
    std::string StringToCast = rc_recur getToken(BaseValue);
    rc_return buildCastExpr(StringToCast, TypeMap.at(BaseValue), CurType);
  }

  if (isCallToTagged(Call, FunctionTags::AddressOf)) {
    // First operand is the type of the value being addressed (should not
    // introduce casts)
    QualifiedType ArgType = deserializeFromLLVMString(Call->getArgOperand(0),
                                                      Model);

    // Second argument is the value being addressed
    llvm::Value *Arg = Call->getArgOperand(1);
    revng_assert(ArgType == TypeMap.at(Arg));

    std::string ArgString = rc_recur getToken(Arg);
    rc_return buildAddressExpr(ArgString);
  }

  if (isCallToTagged(Call, FunctionTags::Parentheses)) {
    std::string Operand0 = rc_recur getToken(Call->getArgOperand(0));
    rc_return addAlwaysParentheses(Operand0);
  }

  if (isCallToTagged(Call, FunctionTags::StructInitializer)) {
    // Struct initializers should be used only to pack together return
    // values of RawFunctionTypes that return multiple values, therefore
    // they must have the same type as the function's return type
    auto *StructTy = cast<llvm::StructType>(Call->getType());
    revng_assert(Call->getFunction()->getReturnType() == StructTy);
    revng_assert(LLVMFunction.getReturnType() == StructTy);
    auto StrucTypeName = getNamedInstanceOfReturnType(ParentPrototype,
                                                      "",
                                                      ThePTMLCBuilder);
    std::string StructInit = addAlwaysParentheses(StrucTypeName);

    // Emit RHS
    llvm::StringRef Separator = "{";
    for (const auto &Arg : Call->args()) {
      StructInit += Separator.str() + " " + rc_recur getToken(Arg);
      Separator = ",";
    }
    StructInit += " }";

    rc_return StructInit;
  }

  if (isCallToTagged(Call, FunctionTags::SegmentRef)) {
    auto *Callee = Call->getCalledFunction();
    const auto &[StartAddress,
                 VirtualSize] = extractSegmentKeyFromMetadata(*Callee);
    model::Segment Segment = Model.Segments().at({ StartAddress, VirtualSize });
    auto Name = Segment.name();

    rc_return ThePTMLCBuilder.getLocationReference(Segment);
  }

  if (isCallToTagged(Call, FunctionTags::Copy))
    rc_return rc_recur getToken(Call->getArgOperand(0));

  if (isCallToTagged(Call, FunctionTags::OpaqueCSVValue)) {
    auto *Callee = Call->getCalledFunction();
    std::string HelperRef = getHelperFunctionLocationReference(Callee,
                                                               ThePTMLCBuilder);
    rc_return rc_recur getCallToken(Call, HelperRef, /*prototype=*/nullptr);
  }

  using PTMLOperator = ptml::PTMLCBuilder::Operator;
  if (isCallToTagged(Call, FunctionTags::UnaryMinus)) {
    auto Operand = Call->getArgOperand(0);
    std::string ToNegate = rc_recur getToken(Operand);
    rc_return ThePTMLCBuilder.getOperator(PTMLOperator::UnaryMinus) + ToNegate;
  }

  if (isCallToTagged(Call, FunctionTags::BinaryNot)) {
    auto Operand = Call->getArgOperand(0);
    std::string ToNegate = rc_recur getToken(Operand);
    rc_return(Operand->getType()->isIntegerTy(1) ?
                ThePTMLCBuilder.getOperator(PTMLOperator::BoolNot) :
                ThePTMLCBuilder.getOperator(PTMLOperator::BinaryNot))
      + ToNegate;
  }

  if (isCallToTagged(Call, FunctionTags::BooleanNot)) {
    auto Operand = Call->getArgOperand(0);
    std::string ToNegate = rc_recur getToken(Operand);
    rc_return ThePTMLCBuilder.getOperator(PTMLOperator::BoolNot) + ToNegate;
  }

  if (isCallToTagged(Call, FunctionTags::StringLiteral)) {
    const auto Operand = Call->getArgOperand(0);
    std::string StringLiteral = rc_recur getToken(Operand);
    rc_return ThePTMLCBuilder.getStringLiteral(StringLiteral).serialize();
  }

  std::string Error = "Cannot get token for custom opcode: "
                      + dumpToString(Call);
  revng_abort(Error.c_str());
  rc_return "";
}

RecursiveCoroutine<std::string>
CCodeGenerator::getIsolatedCallToken(const llvm::CallInst *Call) const {

  // Retrieve the CallEdge
  const auto &[CallEdge, _] = Cache.getCallEdge(Model, Call);
  revng_assert(CallEdge);
  const auto &PrototypePath = Cache.getCallSitePrototype(Model, Call);

  // Construct the callee token (can be a function name or a function
  // pointer)
  std::string CalleeToken;
  if (not isa<llvm::Function>(Call->getCalledOperand())) {
    std::string CalledString = rc_recur getToken(Call->getCalledOperand());
    CalleeToken = addParentheses(CalledString);
  } else {
    if (not CallEdge->DynamicFunction().empty()) {
      // Dynamic Function
      auto &DynFuncID = CallEdge->DynamicFunction();
      auto &DynamicFunc = Model.ImportedDynamicFunctions().at(DynFuncID);
      std::string Location = serializedLocation(ranks::DynamicFunction,
                                                DynamicFunc.key());
      CalleeToken = ThePTMLCBuilder
                      .getTag(ptml::tags::Span, DynamicFunc.name().str())
                      .addAttribute(attributes::Token, tokens::Function)
                      .addAttribute(attributes::ModelEditPath,
                                    getCustomNamePath(DynamicFunc))
                      .addAttribute(attributes::LocationReferences, Location)
                      .serialize();
    } else {
      // Isolated function
      llvm::Function *CalledFunc = Call->getCalledFunction();
      revng_assert(CalledFunc);
      const model::Function *ModelFunc = llvmToModelFunction(Model,
                                                             *CalledFunc);
      revng_assert(ModelFunc);
      CalleeToken = ThePTMLCBuilder
                      .getTag(ptml::tags::Span, ModelFunc->name().str())
                      .addAttribute(attributes::Token, tokens::Function)
                      .addAttribute(attributes::ModelEditPath,
                                    getCustomNamePath(*ModelFunc))
                      .addAttribute(attributes::LocationReferences,
                                    serializedLocation(ranks::Function,
                                                       ModelFunc->key()))
                      .serialize();
    }
  }

  // Build the call expression
  revng_assert(not CalleeToken.empty());
  auto *Prototype = PrototypePath.get();
  rc_return rc_recur getCallToken(Call, CalleeToken, Prototype);
}

RecursiveCoroutine<std::string>
CCodeGenerator::getNonIsolatedCallToken(const llvm::CallInst *Call) const {

  auto *CalledFunc = Call->getCalledFunction();
  revng_assert(CalledFunc and CalledFunc->hasName(),
               "Special functions should all have a name");

  std::string HelperRef = getHelperFunctionLocationReference(CalledFunc,
                                                             ThePTMLCBuilder);
  rc_return rc_recur getCallToken(Call, HelperRef, /*prototype=*/nullptr);
}

static bool shouldGenerateDebugInfoAsPTML(const llvm::Instruction &I) {
  if (!I.getDebugLoc() || !I.getDebugLoc()->getScope())
    return false;

  // If the next instruction in the BB has different DebugLoc, generate the
  // PTML location now.
  auto NextInstr = std::next(I.getIterator());
  if (NextInstr == I.getParent()->end() || !NextInstr->getDebugLoc()
      || NextInstr->getDebugLoc() != I.getDebugLoc())
    return true;
  return false;
}

static std::string addDebugInfo(const llvm::Instruction *I,
                                const std::string &Str,
                                const ptml::PTMLCBuilder &ThePTMLCBuilder) {
  if (shouldGenerateDebugInfoAsPTML(*I))
    return ThePTMLCBuilder.getTag(ptml::tags::Span, Str)
      .addAttribute(ptml::attributes::LocationReferences,
                    I->getDebugLoc()->getScope()->getName())
      .serialize();
  return Str;
}

/// Return the string that represents the given binary operator in C
static const std::string
getBinOpString(const llvm::BinaryOperator *BinOp,
               const ptml::PTMLCBuilder &ThePTMLCBuilder) {
  const Tag Op = [&BinOp, &ThePTMLCBuilder]() {
    bool IsBool = BinOp->getType()->isIntegerTy(1);

    using PTMLOperator = ptml::PTMLCBuilder::Operator;

    switch (BinOp->getOpcode()) {
    case Instruction::Add:
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::Add);
    case Instruction::Sub:
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::Sub);
    case Instruction::Mul:
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::Mul);
    case Instruction::SDiv:
    case Instruction::UDiv:
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::Div);
    case Instruction::SRem:
    case Instruction::URem:
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::Modulo);
    case Instruction::LShr:
    case Instruction::AShr:
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::RShift);
    case Instruction::Shl:
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::LShift);
    case Instruction::And:
      return IsBool ?
               ThePTMLCBuilder.getOperator(PTMLOperator::BoolAnd) :
               ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::And);
    case Instruction::Or:
      return IsBool ?
               ThePTMLCBuilder.getOperator(PTMLOperator::BoolOr) :
               ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::Or);
    case Instruction::Xor:
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::Xor);
    default:
      revng_abort("Unknown const Binary operation");
    }
  }();
  return " " + Op + " ";
}

/// Return the string that represents the given comparison operator in C
static const std::string
getCmpOpString(const llvm::CmpInst::Predicate &Pred,
               const ptml::PTMLCBuilder &ThePTMLCBuilder) {
  using llvm::CmpInst;
  const Tag Op = [&Pred, &ThePTMLCBuilder]() {
    switch (Pred) {
    case CmpInst::ICMP_EQ: ///< equal
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::CmpEq);
    case CmpInst::ICMP_NE: ///< not equal
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::CmpNeq);
    case CmpInst::ICMP_UGT: ///< unsigned greater than
    case CmpInst::ICMP_SGT: ///< signed greater than
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::CmpGt);
    case CmpInst::ICMP_UGE: ///< unsigned greater or equal
    case CmpInst::ICMP_SGE: ///< signed greater or equal
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::CmpGte);
    case CmpInst::ICMP_ULT: ///< unsigned less than
    case CmpInst::ICMP_SLT: ///< signed less than
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::CmpLt);
    case CmpInst::ICMP_ULE: ///< unsigned less or equal
    case CmpInst::ICMP_SLE: ///< signed less or equal
      return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::CmpLte);
    default:
      revng_abort("Unknown comparison operator");
    }
  }();
  return " " + Op + " ";
}

/// Returns a pair of QualifiedTypes to which LHS and RHS has to be casted to
/// for enabling an == or != comparison in C while preserving semantic.
static std::pair<model::QualifiedType, model::QualifiedType>
getCastTargetTypesForEqualityComparisons(model::QualifiedType LHS,
                                         model::QualifiedType RHS) {
  revng_assert(LHS.isScalar() and RHS.isScalar());
  revng_assert(not LHS.isFloat() and not RHS.isFloat());
  revng_assert(*LHS.size() == *RHS.size());

  // If they are the same we don't have to cast anything.
  if (LHS == RHS)
    return { std::move(LHS), std::move(RHS) };

  // If they are both pointer we don't have to cast anything.
  // This could cause UB in case of strict-aliasing, but that's not something
  // that we're trying to guarantee in decompiled code.
  if (LHS.isPointer() and RHS.isPointer())
    return { std::move(LHS), std::move(RHS) };

  // In case only one is a pointer, given that they both have the same size, we
  // can always cast the non-pointer to the pointer-type.
  if (bool LHSIsPointer = LHS.isPointer(); LHSIsPointer != RHS.isPointer()) {
    model::QualifiedType &Pointer = LHSIsPointer ? LHS : RHS;
    return { Pointer, Pointer };
  }

  // At this point we have 2 non-pointer scalar types.
  // Given that we've ruled out Float by assertions, we can just leave them as
  // they are.
  // Even if they mismatch, they have the same size, and in C we'll get an
  // implicit reinterpret cast. This might raise some warning, but we'll deal
  // with those.
  // TODO: this is definitely sloppy, but doing the right thing would require to
  // really think thoroughly about what's the best way to treat casts in
  // general, and we haven't done it yet.
  // At the moment some casts are emitted as ModelCast on the IR others are
  // emitted on the fly during c-code-generation. Until we don't solve that
  // problem systematically, this is a sloppy solution to prevent proliferation
  // of casts, trading off the fact of not having warnings. So in practice this
  // works at the cost of disabling more warnings on decompiled C code. Once
  // we've solved this properly the warning can be re-enabled.
  return { std::move(LHS), std::move(RHS) };
}

RecursiveCoroutine<std::string>
CCodeGenerator::getInstructionToken(const llvm::Instruction *I) const {

  if (isa<llvm::BinaryOperator>(I) or isa<llvm::CmpInst>(I)) {
    const llvm::Value *Op0 = I->getOperand(0);
    const llvm::Value *Op1 = I->getOperand(1);

    std::string Op0Token = rc_recur getToken(Op0);
    std::string Op1Token = rc_recur getToken(Op1);

    const QualifiedType &OpType0 = TypeMap.at(Op0);
    const QualifiedType &OpType1 = TypeMap.at(Op1);

    revng_assert(OpType0.isScalar() and OpType1.isScalar());
    revng_assert(not OpType0.isFloat() and not OpType1.isFloat());
    revng_assert(*OpType0.size() == *OpType1.size());
    uint64_t ByteSize = *OpType0.size();

    if (auto *ICmp = dyn_cast<llvm::ICmpInst>(I)) {
      if (ICmp->isEquality()) {
        // Cast the two operands to a same common type for equality comparison.
        const auto
          &[TargetOp0Type,
            TargetOp1Type] = getCastTargetTypesForEqualityComparisons(OpType0,
                                                                      OpType1);
        Op0Token = buildCastExpr(Op0Token, OpType0, TargetOp0Type);
        Op1Token = buildCastExpr(Op1Token, OpType1, TargetOp0Type);
      } else {
        // If we're not doing eq or neq, we have to make sure that the
        // signedness is compatible, otherwise it would break semantics.
        using model::PrimitiveTypeKind::Signed;
        using model::PrimitiveTypeKind::Unsigned;
        auto ICmpKind = ICmp->isSigned() ? Signed : Unsigned;

        auto TargetType = model::QualifiedType(Model.getPrimitiveType(ICmpKind,
                                                                      ByteSize),
                                               {});
        if (OpType0.isPointer()) {
          Op0Token = buildCastExpr(Op0Token, OpType0, TargetType);
        } else {
          const model::Type *TheType = peelConstAndTypedefs(OpType0)
                                         .UnqualifiedType()
                                         .getConst();
          const auto *Primitive = cast<model::PrimitiveType>(TheType);
          auto CurrentKind = Primitive->PrimitiveKind();
          if (ICmpKind == Signed and CurrentKind != Signed)
            Op0Token = buildCastExpr(Op0Token, OpType0, TargetType);
          if (ICmpKind == Unsigned and CurrentKind == Signed)
            Op0Token = buildCastExpr(Op0Token, OpType0, TargetType);
        }

        if (OpType1.isPointer()) {
          Op1Token = buildCastExpr(Op1Token, OpType1, TargetType);
        } else {
          const model::Type *TheType = peelConstAndTypedefs(OpType1)
                                         .UnqualifiedType()
                                         .getConst();
          const auto *Primitive = cast<model::PrimitiveType>(TheType);
          auto CurrentKind = Primitive->PrimitiveKind();
          if (ICmpKind == Signed and CurrentKind != Signed)
            Op1Token = buildCastExpr(Op1Token, OpType1, TargetType);
          if (ICmpKind == Unsigned and CurrentKind == Signed)
            Op1Token = buildCastExpr(Op1Token, OpType1, TargetType);
        }
      }

    } else {
      const QualifiedType &ResultType = TypeMap.at(I);
      Op0Token = buildCastExpr(Op0Token, OpType0, ResultType);
      Op1Token = buildCastExpr(Op1Token, OpType1, ResultType);
    }

    auto *Bin = dyn_cast<llvm::BinaryOperator>(I);
    auto *Cmp = dyn_cast<llvm::CmpInst>(I);
    revng_assert(Bin or Cmp);
    auto OperatorString = Bin ? getBinOpString(Bin, ThePTMLCBuilder) :
                                getCmpOpString(Cmp->getPredicate(),
                                               ThePTMLCBuilder);

    // TODO: Integer promotion
    rc_return addDebugInfo(I,
                           addParentheses(Op0Token) + OperatorString
                             + addParentheses(Op1Token),
                           ThePTMLCBuilder);
  }

  if (isa<llvm::CastInst>(I) or isa<llvm::FreezeInst>(I)) {

    const llvm::Value *Op = I->getOperand(0);
    std::string ToCast = rc_recur getToken(Op);
    rc_return addDebugInfo(I,
                           buildCastExpr(ToCast, TypeMap.at(Op), TypeMap.at(I)),
                           ThePTMLCBuilder);
  }

  switch (I->getOpcode()) {

  case llvm::Instruction::Call: {
    auto *Call = cast<llvm::CallInst>(I);

    revng_assert(isCallToCustomOpcode(Call) or isCallToIsolatedFunction(Call)
                 or isCallToNonIsolated(Call));

    if (isCallToCustomOpcode(Call))
      rc_return addDebugInfo(I,
                             rc_recur getCustomOpcodeToken(Call),
                             ThePTMLCBuilder);

    if (isCallToIsolatedFunction(Call))
      rc_return addDebugInfo(I,
                             rc_recur getIsolatedCallToken(Call),
                             ThePTMLCBuilder);

    if (isCallToNonIsolated(Call))
      rc_return addDebugInfo(I,
                             rc_recur getNonIsolatedCallToken(Call),
                             ThePTMLCBuilder);

    std::string Error = "Cannot get token for CallInst: " + dumpToString(Call);
    revng_abort(Error.c_str());

    rc_return "";

  } break;

  case llvm::Instruction::Ret: {

    std::string Result = ThePTMLCBuilder
                           .getKeyword(ptml::PTMLCBuilder::Keyword::Return)
                           .serialize();
    if (auto *Ret = llvm::cast<llvm::ReturnInst>(I);
        llvm::Value *ReturnedVal = Ret->getReturnValue())
      Result += " " + rc_recur getToken(ReturnedVal);

    rc_return addDebugInfo(I, Result, ThePTMLCBuilder);

  } break;

  case llvm::Instruction::Unreachable:
    rc_return addDebugInfo(I, "__builtin_trap()", ThePTMLCBuilder);

  case llvm::Instruction::ExtractValue: {

    // Note: ExtractValues at this point should have been already
    // handled when visiting the instruction that generated their
    // struct operand
    auto *ExtractVal = llvm::cast<llvm::ExtractValueInst>(I);
    revng_assert(ExtractVal->getNumIndices() == 1);
    const auto &Idx = ExtractVal->getIndices().back();
    const llvm::Value *AggregateOp = ExtractVal->getAggregateOperand();

    const auto *CallReturnsStruct = llvm::cast<llvm::CallInst>(AggregateOp);
    const llvm::Function *Callee = CallReturnsStruct->getCalledFunction();
    const auto CalleePrototype = Cache.getCallSitePrototype(Model,
                                                            CallReturnsStruct);

    std::string StructFieldRef;
    if (not CalleePrototype.isValid()) {
      // The call returning a struct is a call to a helper function.
      // It must be a direct call.
      revng_assert(Callee);
      StructFieldRef = getReturnStructFieldLocationReference(Callee,
                                                             Idx,
                                                             ThePTMLCBuilder);
    } else {
      const model::Type *CalleeType = CalleePrototype.getConst();
      StructFieldRef = getReturnField(*CalleeType, Idx, Model).str().str();
    }

    rc_return addDebugInfo(I,
                           rc_recur getToken(AggregateOp) + "."
                             + StructFieldRef,
                           ThePTMLCBuilder);

  } break;

  case llvm::Instruction::Select: {

    auto *Select = llvm::cast<llvm::SelectInst>(I);
    std::string Condition = rc_recur getToken(Select->getCondition());
    const llvm::Value *Op1 = Select->getOperand(1);
    const llvm::Value *Op2 = Select->getOperand(2);

    std::string Op1String = rc_recur getToken(Op1);
    std::string Op1Token = buildCastExpr(Op1String,
                                         TypeMap.at(Op1),
                                         TypeMap.at(Select));
    std::string Op2String = rc_recur getToken(Op2);
    std::string Op2Token = buildCastExpr(Op2String,
                                         TypeMap.at(Op2),
                                         TypeMap.at(Select));

    rc_return addDebugInfo(I,
                           addParentheses(Condition) + " ? "
                             + addParentheses(Op1Token) + " : "
                             + addParentheses(Op2Token),
                           ThePTMLCBuilder);

  } break;

  default: {
    std::string Error = "Cannot getToken for llvm::Instruction: "
                        + dumpToString(I);
    revng_abort(Error.c_str());
  }
  }

  std::string Error = "Cannot getToken for llvm::Instruction: "
                      + dumpToString(I);
  revng_abort(Error.c_str());

  rc_return "";
}

RecursiveCoroutine<std::string>
CCodeGenerator::getToken(const llvm::Value *V) const {
  revng_log(Log, "getToken(): " << dumpToString(V));
  LoggerIndent Indent{ Log };
  // If we already have a variable name for this, return it.
  auto It = TokenMap.find(V);
  if (It != TokenMap.end()) {
    revng_assert(isa<llvm::Argument>(V) or isStackFrameDecl(V)
                 or isCallStackArgumentDecl(V) or isLocalVarDecl(V));
    revng_log(Log, "Found!");
    rc_return It->second;
  }

  // We should always have names for stuff that is expected to have a name.
  revng_assert(not isa<llvm::Argument>(V) and not isStackFrameDecl(V)
               and not isCallStackArgumentDecl(V) and not isLocalVarDecl(V));

  if (isCConstant(V))
    rc_return rc_recur getConstantToken(V);

  if (auto *I = dyn_cast<llvm::Instruction>(V))
    rc_return rc_recur getInstructionToken(I);

  std::string Error = "Cannot get token for llvm::Value: ";
  Error += dumpToString(V).c_str();
  revng_abort(Error.c_str());

  rc_return "";
}

RecursiveCoroutine<std::string>
CCodeGenerator::getCallToken(const llvm::CallInst *Call,
                             const llvm::StringRef FuncName,
                             const model::Type *Prototype) const {
  std::string Expression = FuncName.str();
  if (Call->arg_size() == 0) {
    Expression += "()";

  } else {
    llvm::StringRef Separator = "(";
    for (const auto &Arg : Call->args()) {
      Expression += Separator.str() + rc_recur getToken(Arg);
      Separator = ", ";
    }
    Expression += ')';
  }

  rc_return Expression;
}

static bool isStatement(const llvm::Instruction *I) {
  // Return are statements
  if (isa<llvm::ReturnInst>(I))
    return true;

  // Instructions that are not calls are never statement.
  auto *Call = dyn_cast<llvm::CallInst>(I);
  if (not Call)
    return false;

  // If the call returns an aggregate, and it needs a top scope declaration, we
  // have to handle it as if it was an assignment to the local variable declared
  // in the top scope declaration.
  // This is due to the fact that AddAssignmentMarkerPass cannot really inject
  // LocalVariables and Assign/Copy for stuff that has aggregate type on the
  // LLVM IR (because those types are not on the model), so we need to handle it
  // now.
  if (Call->getType()->isAggregateType() and needsTopScopeDeclaration(*Call))
    return true;

  // Calls to Assign and LocalVariable are statemements.
  if (isAssignment(Call) or isLocalVarDecl(Call))
    return true;

  // Calls to isolated functions and helpers that return void are statements.
  // If they don't return void, they are not statements. They are expressions
  // that will be assigned to some local variables in some other assign
  // statements.
  if (isCallToIsolatedFunction(Call) or isCallToNonIsolated(Call))
    return Call->getType()->isVoidTy();

  // Stack frame declarations and call stack arguments declarations are
  // statements.
  if (isStackFrameDecl(Call) or isCallStackArgumentDecl(Call))
    return true;

  return false;
}

void CCodeGenerator::emitBasicBlock(const llvm::BasicBlock *BB) {
  LoggerIndent Indent{ VisitLog };
  revng_log(VisitLog, "|__ Visiting BB " << BB->getName());
  LoggerIndent MoreIndent{ VisitLog };
  revng_log(Log, "--------- BB " << BB->getName());

  for (const Instruction &I : *BB) {
    revng_log(Log, "Analyzing: " << dumpToString(I));

    if (not isStatement(&I)) {
      revng_log(Log, "Ignoring: non-statement instruction");
      continue;
    }

    if (I.getType()->isVoidTy()) {
      revng_assert(isa<llvm::ReturnInst>(I) or isCallToIsolatedFunction(&I)
                   or isCallToNonIsolated(&I) or isAssignment(&I));
      Out << getToken(&I) << ";\n";
      continue;
    }

    // At this point we're left with only CallInst
    auto *Call = cast<llvm::CallInst>(&I);

    // This is a call but it actually needs an assignment to the top scope
    // variable. The top scope variable has not been declared in the IR with
    // LocalVariable, because LocalVariable needs a model type, and aggregates
    // types on the LLVM IR are not on the model.
    if (TopScopeVariables.contains(Call)) {
      revng_assert(Call->getType()->isAggregateType());
      std::string VarName = getVarName(Call);
      revng_assert(not VarName.empty());
      Out << VarName << " "
          << ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::Assign)
          << " " << getToken(Call) << ";\n";
      continue;
    }

    if (isStackFrameDecl(Call)) {
      // Stack frame declaration is a statement, but we've handled explicitly
      // to emit it as the first declaration in this function. So we just
      // assert and go to the next instruction.
      revng_assert(TokenMap.contains(Call));
      continue;
    }

    // Emit variable declaration statements
    if (isLocalVarDecl(Call) or isCallStackArgumentDecl(Call)) {
      // Emit missing local variable declarations
      std::string VarName = createLocalVarDeclName(Call);
      revng_assert(not VarName.empty());
      Out << getNamedCInstance(TypeMap.at(Call), VarName, ThePTMLCBuilder)
          << ";\n";
      continue;
    }

    std::string Error = "Cannot emit statement: ";
    Error += dumpToString(Call).c_str();
    revng_abort(Error.c_str());
  }
}

RecursiveCoroutine<std::string>
CCodeGenerator::buildGHASTCondition(const ExprNode *E) {
  LoggerIndent Indent{ VisitLog };
  revng_log(VisitLog, "|__ Visiting Condition " << E);
  LoggerIndent MoreIndent{ VisitLog };

  using NodeKind = ExprNode::NodeKind;
  switch (E->getKind()) {

  case NodeKind::NK_Atomic: {
    revng_log(VisitLog, "(atomic)");

    // An atomic node holds a reference to the Basic Block that contains the
    // condition used in the conditional expression. In particular, the
    // condition is the value used in the last expression of the basic
    // block.

    // First, emit the BB
    const AtomicNode *Atomic = cast<AtomicNode>(E);
    llvm::BasicBlock *BB = Atomic->getConditionalBasicBlock();
    revng_assert(BB);
    emitBasicBlock(BB);

    // Then, extract the token of the last instruction (must be a
    // conditional branch instruction)
    llvm::Instruction *CondTerminator = BB->getTerminator();
    llvm::BranchInst *Br = cast<llvm::BranchInst>(CondTerminator);
    revng_assert(Br->isConditional());

    // Emit code for x != 0 case with cast.
    auto *I = dyn_cast<llvm::Instruction>(Br->getCondition());
    if (I) {
      auto *Cmp = dyn_cast<llvm::CmpInst>(I);
      const llvm::Value *Op1 = I->getOperand(1);
      if (Cmp and Cmp->getPredicate() == llvm::CmpInst::ICMP_NE
          and dyn_cast<llvm::Constant>(Op1)
          and dyn_cast<llvm::Constant>(Op1)->isZeroValue()) {

        const llvm::Value *Op0 = I->getOperand(0);
        std::string Op0String = rc_recur getToken(Op0);
        model::QualifiedType BoolTy;
        using model::PrimitiveTypeKind::Unsigned;
        BoolTy.UnqualifiedType() = Model.getPrimitiveType(Unsigned, 1);
        rc_return addDebugInfo(I,
                               buildCastExpr(Op0String,
                                             TypeMap.at(Op0),
                                             BoolTy),
                               ThePTMLCBuilder);
      }
    }
    rc_return rc_recur getToken(Br->getCondition());
  } break;

  case NodeKind::NK_Not: {
    revng_log(VisitLog, "(not)");

    const NotNode *N = cast<NotNode>(E);
    ExprNode *Negated = N->getNegatedNode();
    rc_return ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::BoolNot)
      + addAlwaysParentheses(rc_recur buildGHASTCondition(Negated));
  } break;

  case NodeKind::NK_And:
  case NodeKind::NK_Or: {
    revng_log(VisitLog, "(and/or)");

    const BinaryNode *Binary = cast<BinaryNode>(E);

    const auto &[Child1, Child2] = Binary->getInternalNodes();
    std::string Child1Token = rc_recur buildGHASTCondition(Child1);
    std::string Child2Token = rc_recur buildGHASTCondition(Child2);
    using PTMLOperator = ptml::PTMLCBuilder::Operator;
    const Tag &OpToken = E->getKind() == NodeKind::NK_And ?
                           ThePTMLCBuilder.getOperator(PTMLOperator::BoolAnd) :
                           ThePTMLCBuilder.getOperator(PTMLOperator::BoolOr);
    rc_return addAlwaysParentheses(Child1Token) + " " + OpToken.serialize()
      + " " + addAlwaysParentheses(Child2Token);
  } break;

  default:
    revng_abort("Unknown ExprNode kind");
  }
}

RecursiveCoroutine<void> CCodeGenerator::emitGHASTNode(const ASTNode *N) {
  if (N == nullptr)
    rc_return;

  revng_log(VisitLog, "|__ GHAST Node " << N->getID());
  LoggerIndent Indent{ VisitLog };

  auto Kind = N->getKind();
  switch (Kind) {

  case ASTNode::NodeKind::NK_Break: {
    revng_log(VisitLog, "(NK_Break)");

    const BreakNode *Break = llvm::cast<BreakNode>(N);
    using PTMLOperator = ptml::PTMLCBuilder::Operator;
    if (Break->breaksFromWithinSwitch()) {
      revng_assert(not SwitchStateVars.empty()
                   and not SwitchStateVars.back().empty());
      Out << SwitchStateVars.back()
          << " " + ThePTMLCBuilder.getOperator(PTMLOperator::Assign) + " "
               + ThePTMLCBuilder.getTrueTag() + ";\n";
    }
  };
    [[fallthrough]];

  case ASTNode::NodeKind::NK_SwitchBreak: {
    revng_log(VisitLog, "(NK_SwitchBreak)");

    Out << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::Break)
        << ";\n";
  } break;

  case ASTNode::NodeKind::NK_Continue: {
    revng_log(VisitLog, "(NK_Continue)");

    const ContinueNode *Continue = cast<ContinueNode>(N);

    // Print the condition computation code of the if statement.
    if (Continue->hasComputation()) {
      IfNode *ComputationIfNode = Continue->getComputationIfNode();
      rc_recur buildGHASTCondition(ComputationIfNode->getCondExpr());
    }

    // Actually print the continue statement only if the continue is not
    // implicit (i.e. it is not the last statement of the loop).
    if (not Continue->isImplicit())
      Out << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::Continue)
          << ";\n";
  } break;

  case ASTNode::NodeKind::NK_Code: {
    revng_log(VisitLog, "(NK_Code)");

    const CodeNode *Code = cast<CodeNode>(N);
    llvm::BasicBlock *BB = Code->getOriginalBB();
    revng_assert(BB != nullptr);
    emitBasicBlock(BB);
  } break;

  case ASTNode::NodeKind::NK_If: {
    revng_log(VisitLog, "(NK_If)");

    const IfNode *If = cast<IfNode>(N);
    std::string CondExpr = rc_recur buildGHASTCondition(If->getCondExpr());
    // "If" expression
    // TODO: possibly cast the CondExpr if it's not convertible to boolean?
    revng_assert(not CondExpr.empty());
    Out << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::If)
        << " (" + CondExpr + ") ";
    {
      Scope TheScope(Out);
      // "Then" expression (always emitted)
      if (nullptr == If->getThen())
        Out << ThePTMLCBuilder.getLineComment("Empty");
      else
        rc_recur emitGHASTNode(If->getThen());
    }

    // "Else" expression (optional)
    if (If->hasElse()) {
      Out << " " + ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::Else)
               + " ";
      Scope TheScope(Out);
      rc_recur emitGHASTNode(If->getElse());
    }
    Out << "\n";
  } break;

  case ASTNode::NodeKind::NK_Scs: {
    revng_log(VisitLog, "(NK_Scs)");

    const ScsNode *LoopBody = cast<ScsNode>(N);

    // Calculate the string of the condition
    // TODO: possibly cast the CondExpr if it's not convertible to boolean?
    std::string CondExpr = ThePTMLCBuilder.getTrueTag().serialize();
    if (LoopBody->isDoWhile() or LoopBody->isWhile()) {
      const IfNode *LoopCondition = LoopBody->getRelatedCondition();
      revng_assert(LoopCondition);

      // Retrieve the expression of the condition as well as emitting its
      // associated basic block
      CondExpr = rc_recur buildGHASTCondition(LoopCondition->getCondExpr());
      revng_assert(not CondExpr.empty());
    }

    if (LoopBody->isDoWhile())
      Out << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::Do) << " ";
    else
      Out << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::While)
               + " (" + CondExpr + ") ";

    revng_assert(LoopBody->hasBody());
    {
      Scope TheScope(Out);
      rc_recur emitGHASTNode(LoopBody->getBody());
    }

    if (LoopBody->isDoWhile())
      Out << " "
               + ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::While)
               + " (" + CondExpr + ");";
    Out << "\n";

  } break;

  case ASTNode::NodeKind::NK_List: {
    revng_log(VisitLog, "(NK_List)");

    const SequenceNode *Seq = cast<SequenceNode>(N);
    for (const ASTNode *Child : Seq->nodes())
      rc_recur emitGHASTNode(Child);

  } break;

  case ASTNode::NodeKind::NK_Switch: {
    revng_log(VisitLog, "(NK_Switch)");

    const SwitchNode *Switch = cast<SwitchNode>(N);

    // If needed, print the declaration of the switch state variable, which
    // is used by nested switches inside loops to break out of the loop
    if (Switch->needsStateVariable()) {
      revng_assert(Switch->needsLoopBreakDispatcher());
      StringToken NewVarName = NameGenerator.nextSwitchStateVar();
      std::string
        SwitchStateVar = getVariableLocationReference(NewVarName,
                                                      ModelFunction,
                                                      ThePTMLCBuilder);
      SwitchStateVars.push_back(std::move(SwitchStateVar));
      using PTMLOperator = ptml::PTMLCBuilder::Operator;
      Out << ThePTMLCBuilder.tokenTag("bool", ptml::c::tokens::Type) << " "
          << getVariableLocationDefinition(NewVarName,
                                           ModelFunction,
                                           ThePTMLCBuilder)
          << " " + ThePTMLCBuilder.getOperator(PTMLOperator::Assign) + " "
               + ThePTMLCBuilder.getFalseTag() + ";\n";
    }

    // Generate the condition of the switch
    StringToken SwitchVarToken;
    model::QualifiedType SwitchVarType;
    llvm::Value *SwitchVar = Switch->getCondition();
    if (SwitchVar) {
      // If the switch is not weaved we need to print the instructions in
      // the basic block before it.
      if (not Switch->isWeaved()) {
        llvm::BasicBlock *BB = Switch->getOriginalBB();
        revng_assert(BB != nullptr); // This is not a switch dispatcher.
        emitBasicBlock(BB);
      }
      std::string SwitchVarString = getToken(SwitchVar);
      SwitchVarToken = SwitchVarString;
      SwitchVarType = TypeMap.at(SwitchVar);
    } else {
      revng_assert(Switch->getOriginalBB() == nullptr);
      revng_assert(!LoopStateVar.empty());
      // This switch does not come from an instruction: it's a dispatcher
      // for the loop state variable
      SwitchVarToken = LoopStateVar;

      // TODO: finer decision on the type of the loop state variable
      using model::PrimitiveTypeKind::Unsigned;
      SwitchVarType.UnqualifiedType() = Model.getPrimitiveType(Unsigned, 8);
    }
    revng_assert(not SwitchVarToken.empty());

    if (not SwitchVarType.is(model::TypeKind::PrimitiveType)) {
      model::QualifiedType BoolTy;
      // TODO: finer decision on how to cast structs used in a switch
      using model::PrimitiveTypeKind::Unsigned;
      BoolTy.UnqualifiedType() = Model.getPrimitiveType(Unsigned, 8);

      SwitchVarToken = buildCastExpr(SwitchVarToken, SwitchVarType, BoolTy);
    }

    // Generate the switch statement
    Out << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::Switch)
             + " ("
        << SwitchVarToken << ") ";
    {
      Scope TheScope(Out);
      using PTMLKeyword = ptml::PTMLCBuilder::Keyword;

      // Generate the body of the switch (except for the default)
      for (const auto &[Labels, CaseNode] : Switch->cases_const_range()) {
        revng_assert(not Labels.empty());
        // Generate the case label(s) (multiple case labels might share the
        // same body)
        for (uint64_t CaseVal : Labels) {
          Out << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::Case)
                   + " ";
          if (SwitchVar) {
            llvm::Type *SwitchVarT = SwitchVar->getType();
            auto *IntType = cast<llvm::IntegerType>(SwitchVarT);
            auto *CaseConst = llvm::ConstantInt::get(IntType, CaseVal);
            // TODO: assigned the signedness based on the signedness of the
            // condition
            Out << ThePTMLCBuilder.getNumber(CaseConst->getValue());
          } else {
            Out << ThePTMLCBuilder.getNumber(CaseVal);
          }
          Out << ":\n";
        }

        {
          Scope InnerScope(Out);
          // Generate the case body
          rc_recur emitGHASTNode(CaseNode);
        }
        Out << " " + ThePTMLCBuilder.getKeyword(PTMLKeyword::Break) + ";\n";
      }

      // Generate the default case if it exists
      if (auto *Default = Switch->getDefault()) {
        Out << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::Default)
            << ":\n";
        {
          Scope TheScope(Out);
          rc_recur emitGHASTNode(Default);
        }
        Out << " " + ThePTMLCBuilder.getKeyword(PTMLKeyword::Break) + ";\n";
      }
    }
    Out << "\n";

    // If the switch needs a loop break dispatcher, reset the associated
    // state variable before emitting the switch statement.
    if (Switch->needsLoopBreakDispatcher()) {
      revng_assert(not SwitchStateVars.empty()
                   and not SwitchStateVars.back().empty());
      Out << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::If) + " ("
               + SwitchStateVars.back() + ")";
      {
        auto Scope = ThePTMLCBuilder.getScope(ptml::PTMLCBuilder::Scopes::Scope)
                       .scope(Out, true);
        auto IndentScope = Out.scope();
        Out << ThePTMLCBuilder.getKeyword(ptml::PTMLCBuilder::Keyword::Break)
                 + ";";
      }
      Out << "\n";
    }

    // If we're done with a switch that generates a state variable to break
    // out of loops, pop it from the stack.
    if (Switch->needsStateVariable()) {
      revng_assert(Switch->needsLoopBreakDispatcher());
      SwitchStateVars.pop_back();
    }

  } break;

  case ASTNode::NodeKind::NK_Set: {
    revng_log(VisitLog, "(NK_Set)");

    const SetNode *Set = cast<SetNode>(N);
    unsigned StateValue = Set->getStateVariableValue();
    revng_assert(!LoopStateVar.empty());

    // Print an assignment to the loop state variable. This is an artificial
    // variable introduced by the GHAST to enable executing certain pieces
    // of code based on which control-flow branch was taken. This, for
    // example, can be used to jump to the middle of a loop
    // instead of at the start, without emitting gotos.
    Out << LoopStateVar << " "
        << ThePTMLCBuilder.getOperator(ptml::PTMLCBuilder::Operator::Assign)
        << " " << StateValue << ";\n";
  } break;
  }

  rc_return;
}

static std::string getModelArgIdentifier(const model::Type *ModelFunctionType,
                                         const llvm::Argument &Argument) {
  const llvm::Function *LLVMFunction = Argument.getParent();
  unsigned ArgNo = Argument.getArgNo();

  if (auto *RFT = dyn_cast<model::RawFunctionType>(ModelFunctionType)) {
    auto NumModelArguments = RFT->Arguments().size();
    revng_assert(ArgNo <= NumModelArguments + 1);
    revng_assert(LLVMFunction->arg_size() == NumModelArguments
                 or (RFT->StackArgumentsType().UnqualifiedType().isValid()
                     and (LLVMFunction->arg_size() == NumModelArguments + 1)));
    if (ArgNo < NumModelArguments) {
      return std::next(RFT->Arguments().begin(), ArgNo)->name().str().str();
    } else {
      return "stack_args";
    }
  } else if (auto *CFT = dyn_cast<model::CABIFunctionType>(ModelFunctionType)) {
    revng_assert(LLVMFunction->arg_size() == CFT->Arguments().size());
    revng_assert(ArgNo < CFT->Arguments().size());
    return CFT->Arguments().at(ArgNo).name().str().str();
  }
  revng_abort("Unexpected function type");

  return "";
}

void CCodeGenerator::emitFunction(bool NeedsLocalStateVar,
                                  InlineableTypesMap &StackTypes) {
  revng_log(Log, "========= Emitting Function " << LLVMFunction.getName());
  revng_log(VisitLog, "========= Function " << LLVMFunction.getName());
  LoggerIndent Indent{ VisitLog };

  auto FunctionTagScope = ThePTMLCBuilder
                            .getScope(ptml::PTMLCBuilder::Scopes::FunctionBody)
                            .scope(Out);

  // Print function's prototype
  printFunctionPrototype(ParentPrototype,
                         ModelFunction,
                         Out,
                         ThePTMLCBuilder,
                         Model,
                         true);

  // Set up the argument identifiers to be used in the function's body.
  for (const auto &Arg : LLVMFunction.args()) {
    std::string ArgString = getModelArgIdentifier(&ParentPrototype, Arg);
    TokenMap[&Arg] = getArgumentLocationReference(ArgString,
                                                  ModelFunction,
                                                  ThePTMLCBuilder);
  }

  // Print the function body
  Out << " ";
  {
    Scope BraceScope(Out, ptml::c::scopes::FunctionBody);

    // We expect just one stack type definition.
    bool IsStackDefined = false;

    // Declare the local variable representing the stack frame
    if (ModelFunction.StackFrameType().isValid()) {
      revng_log(Log, "Stack Frame Declaration");
      const auto &IsStackFrameDecl = [](const llvm::Instruction &I) {
        return isStackFrameDecl(&I);
      };
      auto It = llvm::find_if(llvm::instructions(LLVMFunction),
                              IsStackFrameDecl);
      if (It != llvm::instructions(LLVMFunction).end()) {
        const auto *Call = &cast<llvm::CallInst>(*It);
        std::string VarName = createTopScopeVarDeclName(Call);
        revng_assert(not VarName.empty());
        auto *TheType = ModelFunction.StackFrameType().getConst();
        // This will contain the stack types that we can inline, since
        // there could be a stack type that is being used somewhere else,
        // so we do not want to inline it.
        auto TheStackTypes = StackTypes.at(&ModelFunction);
        if (TheStackTypes.contains(TheType) and !IsStackDefined) {
          IsStackDefined = true;
          QualifiedTypeNameMap AdditionalTypeNames;
          // For all nested types within stack definition we print forward
          // declarations.
          for (auto *Type : TheStackTypes) {
            revng_assert(isCandidateForInline(Type));
            printForwardDeclaration(*Type, Out, ThePTMLCBuilder);
          }
          printDefinition(Log,
                          *cast<model::StructType>(TheType),
                          Out,
                          ThePTMLCBuilder,
                          TheStackTypes,
                          AdditionalTypeNames,
                          Model,
                          VarName);
        } else {
          Out << getNamedCInstance(TypeMap.at(Call), VarName, ThePTMLCBuilder)
              << ";\n";
        }
      } else {

        revng_log(Log,
                  "WARNING: function with valid stack type has no stack "
                  "declaration: "
                    << LLVMFunction.getName());
      }
    }

    // Declare all variables that have the entire function as a scope
    if (not TopScopeVariables.empty()) {

      revng_log(Log, "Top-Scope Declarations");
      for (const llvm::Instruction *VarToDeclare : TopScopeVariables) {
        revng_log(Log, "VarToDeclare: " + dumpToString(VarToDeclare));

        std::string VarName = createTopScopeVarDeclName(VarToDeclare);
        revng_assert(not VarName.empty());

        auto VarTypeIt = TypeMap.find(VarToDeclare);
        if (VarTypeIt != TypeMap.end()) {
          Out << getNamedCInstance(TypeMap.at(VarToDeclare),
                                   VarName,
                                   ThePTMLCBuilder)
              << ";\n";
        } else {
          // The only types that are allowed to be missing from the TypeMap
          // are LLVM aggregates returned by RawFunctionTypes or by helpers
          auto *Call = llvm::cast<CallInst>(VarToDeclare);
          if (const auto &Prototype = Cache.getCallSitePrototype(Model, Call);
              Prototype.isValid() and not Prototype.empty()) {
            const auto *FunctionType = Prototype.getConst();
            Out << getNamedInstanceOfReturnType(*FunctionType,
                                                VarName,
                                                ThePTMLCBuilder)
                << ";\n";
          } else {
            auto *CalledFunction = Call->getCalledFunction();
            revng_assert(CalledFunction);
            Out << getReturnTypeLocationReference(CalledFunction,
                                                  ThePTMLCBuilder)
                << " " << VarName << ";\n";
          }
        }
      }
      revng_log(Log, "End of Top-Scope Declarations");
    }

    // Emit a declaration for the loop state variable, which is used to
    // redirect control flow inside loops (e.g. if we want to jump in the
    // middle of a loop during a certain iteration)
    if (NeedsLocalStateVar)
      Out << ThePTMLCBuilder.tokenTag("uint64_t", ptml::c::tokens::Type) << " "
          << LoopStateVarDeclaration << ";\n";

    // Recursively print the body of this function
    emitGHASTNode(GHAST.getRoot());
  }

  Out << "\n";
}

static std::string decompileFunction(FunctionMetadataCache &Cache,
                                     const llvm::Function &LLVMFunc,
                                     const ASTTree &CombedAST,
                                     const Binary &Model,
                                     const InstrSetVec &TopScopeVariables,
                                     bool NeedsLocalStateVar,
                                     InlineableTypesMap &StackTypes) {
  std::string Result;

  llvm::raw_string_ostream Out(Result);
  ptml::PTMLCBuilder ThePTMLCBuilder;

  CCodeGenerator Backend(Cache,
                         Model,
                         LLVMFunc,
                         CombedAST,
                         TopScopeVariables,
                         Out,
                         ThePTMLCBuilder);
  Backend.emitFunction(NeedsLocalStateVar, StackTypes);
  Out.flush();

  return Result;
}

/// Visit the node and all its children recursively, checking if a loop
/// variable is needed.
// TODO: This could be precomputed and attached to the SCS node in the GHAST.
static RecursiveCoroutine<bool> needsLoopVar(ASTNode *N) {
  if (N == nullptr)
    rc_return false;

  auto Kind = N->getKind();
  switch (Kind) {

  case ASTNode::NodeKind::NK_Break:
  case ASTNode::NodeKind::NK_SwitchBreak:
  case ASTNode::NodeKind::NK_Continue:
  case ASTNode::NodeKind::NK_Code:
    rc_return false;
    break;

  case ASTNode::NodeKind::NK_If: {
    IfNode *If = cast<IfNode>(N);

    if (nullptr != If->getThen())
      if (rc_recur needsLoopVar(If->getThen()))
        rc_return true;

    if (If->hasElse())
      if (rc_recur needsLoopVar(If->getElse()))
        rc_return true;

    rc_return false;
  } break;

  case ASTNode::NodeKind::NK_Scs: {
    ScsNode *LoopBody = cast<ScsNode>(N);
    rc_return rc_recur needsLoopVar(LoopBody->getBody());
  } break;

  case ASTNode::NodeKind::NK_List: {
    SequenceNode *Seq = cast<SequenceNode>(N);
    for (ASTNode *Child : Seq->nodes())
      if (rc_recur needsLoopVar(Child))
        rc_return true;

    rc_return false;
  } break;

  case ASTNode::NodeKind::NK_Switch: {
    SwitchNode *Switch = cast<SwitchNode>(N);
    llvm::Value *SwitchVar = Switch->getCondition();

    if (not SwitchVar)
      rc_return true;

    for (const auto &[Labels, CaseNode] : Switch->cases())
      if (rc_recur needsLoopVar(CaseNode))
        rc_return true;

    if (auto *Default = Switch->getDefault())
      if (rc_recur needsLoopVar(Default))
        rc_return true;

    rc_return false;
  } break;

  case ASTNode::NodeKind::NK_Set: {
    rc_return true;
  } break;
  }
}

static bool hasLoopDispatchers(const ASTTree &GHAST) {
  return needsLoopVar(GHAST.getRoot());
}

static InstrSetVec collectTopScopeVariables(const llvm::Function &F) {
  InstrSetVec TopScopeVars;
  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {
      if (auto *Call = dyn_cast<llvm::CallInst>(&I)) {
        // All the others have already been promoted to LocalVariable Copy and
        // Assign.
        if (not Call->getType()->isAggregateType())
          continue;

        if (isCallToNonIsolated(Call) or isCallToIsolatedFunction(Call)) {
          const auto *Called = Call->getCalledFunction();
          revng_assert(not Called or not Called->isTargetIntrinsic());

          if (needsTopScopeDeclaration(*Call))
            TopScopeVars.insert(Call);
        }
      }
    }
  }

  return TopScopeVars;
}

using Container = revng::pipes::DecompiledCCodeInYAMLStringMap;
void decompile(FunctionMetadataCache &Cache,
               llvm::Module &Module,
               const model::Binary &Model,
               Container &DecompiledFunctions) {
  TypeInlineHelper TheTypeInlineHelper(Model);

  // Get all Stack types and all the inlinable types reachable from it,
  // since we want to emit forward declarations for all of them.
  auto StackTypes = TheTypeInlineHelper.findStackTypesPerFunction(Model);

  for (llvm::Function &F : FunctionTags::Isolated.functions(&Module)) {

    if (F.empty())
      continue;

    // TODO: this will eventually become a GHASTContainer for revng pipeline
    ASTTree GHAST;

    // Generate the GHAST and beautify it.
    {
      restructureCFG(F, GHAST);
      // TODO: beautification should be optional, but at the moment it's not
      // truly so (if disabled, things crash). We should strive to make it
      // optional for real.
      beautifyAST(F, GHAST);
    }

    if (Log.isEnabled()) {
      std::string ASTFileName = F.getName().str()
                                + "GHAST-during-c-codegen.dot";
      GHAST.dumpASTOnFile(ASTFileName.c_str());
    }

    // Generated C code for F
    auto TopScopeVariables = collectTopScopeVariables(F);
    auto NeedsLoopStateVar = hasLoopDispatchers(GHAST);
    std::string CCode = decompileFunction(Cache,
                                          F,
                                          GHAST,
                                          Model,
                                          TopScopeVariables,
                                          NeedsLoopStateVar,
                                          StackTypes);

    // Push the C code into
    MetaAddress Key = getMetaAddressMetadata(&F, "revng.function.entry");
    DecompiledFunctions.insert_or_assign(Key, std::move(CCode));
  }
}
