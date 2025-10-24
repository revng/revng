//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
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
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Progress.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/Backend/DecompileFunction.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/HeadersGeneration/Options.h"
#include "revng/InitModelTypes/InitModelTypes.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Helpers.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Model/PrimitiveKind.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/Segment.h"
#include "revng/Model/StructDefinition.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/PTML/CBuilder.h"
#include "revng/PTML/Constants.h"
#include "revng/PTML/IndentedOstream.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/DebugInfoHelpers.h"
#include "revng/Pipes/Ranks.h"
#include "revng/RestructureCFG/ASTNode.h"
#include "revng/RestructureCFG/ASTNodeUtils.h"
#include "revng/RestructureCFG/ASTTree.h"
#include "revng/RestructureCFG/BeautifyGHAST.h"
#include "revng/RestructureCFG/RestructureCFG.h"
#include "revng/Support/Assert.h"
#include "revng/Support/DecompilationHelpers.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TypeNames/LLVMTypeNames.h"
#include "revng/TypeNames/ModelCBuilder.h"
#include "revng/Yield/PTML.h"

#include "ALAPVariableDeclaration.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;

using llvm::BasicBlock;
using llvm::CallInst;
using llvm::Instruction;
using llvm::raw_ostream;
using llvm::StringRef;

using model::Binary;
using model::CABIFunctionDefinition;
using model::RawFunctionDefinition;

using pipeline::locationString;
using ptml::Tag;
namespace ranks = revng::ranks;
namespace attributes = ptml::attributes;
namespace tokens = ptml::c::tokens;

using tokenDefinition::types::StringToken;

using TokenMapT = std::map<const llvm::Value *, std::string>;
using ModelTypesMap = std::map<const llvm::Value *,
                               const model::UpcastableType>;

static Logger<> Log{ "c-backend" };
static Logger<> VisitLog{ "c-backend-visit-order" };

static bool isStackFrameDecl(const llvm::Value *I) {
  auto *Call = dyn_cast_or_null<llvm::CallInst>(I);
  if (not Call)
    return false;

  auto *Callee = getCalledFunction(Call);
  if (not Callee)
    return false;

  return Callee->getName().startswith("revng_stack_frame");
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
         or isCallToTagged(I, FunctionTags::OpaqueExtractValue)
         or isCallToTagged(I, FunctionTags::StructInitializer)
         or isCallToTagged(I, FunctionTags::SegmentRef)
         or isCallToTagged(I, FunctionTags::UnaryMinus)
         or isCallToTagged(I, FunctionTags::BinaryNot)
         or isCallToTagged(I, FunctionTags::BooleanNot)
         or isCallToTagged(I, FunctionTags::StringLiteral)
         or isCallToTagged(I, FunctionTags::Comment);
}

static bool isCConstant(const llvm::Value *V) {
  return isa<llvm::Constant>(V)
         or isCallToTagged(V, FunctionTags::LiteralPrintDecorator);
}

static std::string addAlwaysParentheses(llvm::StringRef Expr) {
  return std::string("(") + Expr.str() + ")";
}

static std::string get128BitIntegerHexConstant(llvm::APInt Value,
                                               const ptml::ModelCBuilder &B) {
  revng_assert(Value.getBitWidth() > 64);
  revng_assert(Value.getBitWidth() <= 128);
  using PTMLOperator = ptml::CBuilder::Operator;

  auto U128 = model::PrimitiveType::makeUnsigned(16);
  std::string Cast = addAlwaysParentheses(B.getTypeName(*U128));

  if (Value.isZero())
    return addAlwaysParentheses(Cast + " " + B.getNumber(0));

  // In C, even if you can have 128-bit variables, you cannot have 128-bit
  // literals, so we need this hack to assign a big constant value to a
  // 128-bit variable.
  llvm::APInt HighBits = Value.getHiBits(Value.getBitWidth() - 64);
  llvm::APInt LowBits = Value.getLoBits(64);
  bool NeedsOr = not HighBits.isZero() and not LowBits.isZero();

  std::string CompositeConstant = Cast + " ";

  if (not HighBits.isZero()) {
    StringToken HighBitsString;
    HighBits.toString(HighBitsString,
                      /*radix=*/16,
                      /*signed=*/false,
                      /*formatAsCLiteral=*/true);

    auto HighConst = B.getConstantTag(HighBitsString) + " "
                     + B.getOperator(PTMLOperator::LShift) + " "
                     + B.getNumber(64);

    CompositeConstant += HighConst;
  }

  if (NeedsOr)
    CompositeConstant += " " + B.getOperator(PTMLOperator::Or) + " ";

  if (not LowBits.isZero()) {
    StringToken LowBitsString;
    LowBits.toString(LowBitsString,
                     /*radix=*/16,
                     /*signed=*/false,
                     /*formatAsCLiteral=*/true);
    CompositeConstant += B.getConstantTag(LowBitsString).toString();
  }
  return addAlwaysParentheses(CompositeConstant);
}

static std::string hexLiteral(const llvm::ConstantInt *Int,
                              const ptml::ModelCBuilder &B) {
  StringToken Formatted;
  if (Int->getBitWidth() <= 64) {
    Int->getValue().toString(Formatted,
                             /*radix*/ 16,
                             /*signed*/ false,
                             /*formatAsCLiteral*/ true);
    return Formatted.str().str();
  }
  return get128BitIntegerHexConstant(Int->getValue(), B);
}

static std::string charLiteral(const llvm::ConstantInt *Int) {
  revng_assert(Int->getValue().getBitWidth() == 8);
  const auto LimitedValue = Int->getLimitedValue(0xffu);
  const auto CharValue = static_cast<char>(LimitedValue);

  std::string Escaped;
  llvm::raw_string_ostream EscapeCStream(Escaped);
  EscapeCStream.write_escaped(std::string(&CharValue, 1));

  return "'" + std::move(Escaped) + "'";
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
  const model::TypeDefinition &Prototype;
  /// The (combed) control flow AST
  const ASTTree &GHAST;

  /// A map that associates to each ASTNode, a set of variables to be declared
  /// in that scope, with a specific order.
  /// A variable is represented by a CallInst to LocalVariable
  const ASTVarDeclMap &VariablesToDeclare;

  /// A map containing a model type for each LLVM value in the function
  const ModelTypesMap TypeMap;

  /// Helper for outputting the decompiled C code
  ptml::ModelCBuilder &B;

  /// Name of the local variable used to break out of loops from within nested
  /// switches
  std::vector<std::string> SwitchStateVars;

  /// \note This class handles an individual function, but still needs a
  ///       reference to all of ControlFlowGraphCache, so we can call
  ///       getCallEdge. This is not nice, since, if we look into stuff
  ///       concerning other functions, we can create serious invalidation
  ///       issues.
  ///       However, right now we just call getCallEdge which is safe.
  ///       A better approach would be to have a ControlFlowGraphCache provide
  ///       us with a function-specific subject that enables us to use
  ///       getCallEdge.
  ControlFlowGraphCache &Cache;

  /// Stateful generator for variable names
  ptml::ModelCBuilder::VariableNameBuilder VariableNameBuilder;

  /// Keep track of the names associated with function arguments, and local
  /// variables. In the past it also kept track of intermediate expressions, but
  /// with the new design all the tokens corresponding to instructions that
  /// don't represent local variables are recomputed every time.
  TokenMapT TokenMap;

private:
  /// Name of the local variable used to break out from loops
  std::string LoopStateVar;
  std::string LoopStateVarDeclaration;

  /// `dumpToString` optimization helper.
  llvm::ModuleSlotTracker ModuleSlotTracker;

private:
  /// Emission of parentheses may change whether the OPRP is enabled or not
  bool IsOperatorPrecedenceResolutionPassEnabled = false;

public:
  CCodeGenerator(ControlFlowGraphCache &Cache,
                 const Binary &Model,
                 const llvm::Function &LLVMFunction,
                 const ASTTree &GHAST,
                 const ASTVarDeclMap &VarToDeclare,
                 ptml::ModelCBuilder &B) :
    Model(Model),
    LLVMFunction(LLVMFunction),
    ModelFunction(*llvmToModelFunction(Model, LLVMFunction)),
    Prototype(*Model.prototypeOrDefault(ModelFunction.prototype())),
    GHAST(GHAST),
    VariablesToDeclare(VarToDeclare),
    TypeMap(initModelTypes(LLVMFunction,
                           &ModelFunction,
                           Model,
                           /* PointersOnly = */ false)),
    B(B),
    SwitchStateVars(),
    Cache(Cache),
    VariableNameBuilder(B.makeLocalVariableNameBuilder(ModelFunction)),
    ModuleSlotTracker(LLVMFunction.getParent(),
                      /* ShouldInitializeAllMetadata = */ false) {

    if (Log.isEnabled())
      ModuleSlotTracker.incorporateFunction(LLVMFunction);

    // TODO: don't use a global loop state variable
    const auto &Configuration = B.NameBuilder.Configuration;
    llvm::StringRef LoopStateVariable = Configuration.LoopStateVariableName();
    auto [Definition, Reference] = B.getReservedVariableTags(ModelFunction,
                                                             LoopStateVariable);
    LoopStateVarDeclaration = std::move(Definition);
    LoopStateVar = std::move(Reference);

    if (LLVMFunction.getMetadata(ExplicitParenthesesMDName))
      IsOperatorPrecedenceResolutionPassEnabled = true;
  }

  void emitFunction(bool NeedsLocalStateVar);

private:
  /// Visit a GHAST node and all its children recursively, emitting BBs
  /// and control flow statements in the process.
  RecursiveCoroutine<void> emitGHASTNode(const ASTNode *Node);

  /// Recursively build a C string representing the condition contained
  /// in an ExprNode (which might be composed by one or more subexpressions).
  /// An additional parameter is used to decide whether the basic block
  /// associated to an atomic or compare node should be emitted on-the-fly.
  RecursiveCoroutine<std::string> buildGHASTCondition(const ExprNode *E,
                                                      bool EmitBB);

  RecursiveCoroutine<std::string>
  makeLoopCondition(const IfNode *LoopCondition) {
    revng_assert(LoopCondition);

    // Retrieve the expression of the condition as well as emitting its
    // associated basic block
    bool EmitBB = not LoopCondition->isWeaved();
    rc_return rc_recur buildGHASTCondition(LoopCondition->getCondExpr(),
                                           EmitBB);
  }

  /// Serialize a basic block into a series of C statements.
  void emitBasicBlock(const BasicBlock *BB, bool EmitReturn);

private:
  RecursiveCoroutine<std::string> getToken(const llvm::Value *V) const;

  RecursiveCoroutine<std::string>
  getCallToken(const llvm::CallInst *Call,
               const llvm::StringRef FuncName,
               const model::TypeDefinition *Prototype) const;

  RecursiveCoroutine<std::string> getConstantToken(const llvm::Value *V) const;

  RecursiveCoroutine<std::string>
  getInstructionToken(const llvm::Instruction *I) const;

  RecursiveCoroutine<std::string>
  getCustomOpcodeToken(const llvm::CallInst *C) const;

  RecursiveCoroutine<std::string>
  getModelGEPToken(const llvm::CallInst *C) const;

  std::string getIsolatedFunctionToken(const llvm::Function *F) const;

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
                            const model::Type &SrcType,
                            const model::Type &DestType) const;

private:
  std::string createStackFrameVarDeclName(const llvm::Instruction *I) {
    revng_assert(isStackFrameDecl(I));
    revng_assert(not TokenMap.contains(I));

    const auto &Configuration = Model.Configuration().Naming();
    llvm::StringRef VarName = Configuration.StackFrameVariableName();
    auto [Definition, Reference] = B.getReservedVariableTags(ModelFunction,
                                                             VarName);
    TokenMap[I] = std::move(Reference);
    return std::move(Definition);
  }

  std::string createLocalVarDeclName(const llvm::Instruction *I) {
    revng_assert(isLocalVarDecl(I) or isCallStackArgumentDecl(I)
                   or isArtificialAggregateLocalVarDecl(I)
                   or isHelperAggregateLocalVarDecl(I),
                 "This instruction is not a local variable!");

    SortedVector<MetaAddress> UserAddressList;
    for (const llvm::Value *UserValue : I->users()) {
      if (const auto *User = llvm::dyn_cast<llvm::Instruction>(UserValue)) {
        if (std::optional MaybeAddress = revng::tryExtractAddress(*User)) {
          UserAddressList.emplace(std::move(MaybeAddress.value()));

        } else {
          // Found a user without debug information,
          // discard current variable.
          // TODO: is there a way to handle this case more gracefully?
          UserAddressList.clear();
          break;
        }
      }
    }

    // This may override the entry for I, if I belongs to a "duplicated"
    // BasicBlock that is reachable from many paths on the GHAST.
    auto [Definition, Reference] = B.getVariableTags(VariableNameBuilder,
                                                     UserAddressList);
    TokenMap[I] = std::move(Reference);
    return std::move(Definition);
  }

  std::string getVarName(const llvm::Instruction *I) const {
    revng_assert(isStackFrameDecl(I) or isLocalVarDecl(I)
                 or isCallStackArgumentDecl(I)
                 or isArtificialAggregateLocalVarDecl(I)
                 or isHelperAggregateLocalVarDecl(I));
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
  using PTMLOperator = ptml::CBuilder::Operator;
  return B.getOperator(PTMLOperator::PointerDereference) + addParentheses(Expr);
}

std::string CCodeGenerator::buildAddressExpr(llvm::StringRef Expr) const {
  return B.getOperator(ptml::CBuilder::Operator::AddressOf)
         + addParentheses(Expr);
}

std::string CCodeGenerator::buildCastExpr(StringRef ExprToCast,
                                          const model::Type &SrcType,
                                          const model::Type &DestType) const {
  if (SrcType == DestType)
    return ExprToCast.str();

  if (*SrcType.skipTypedefs() != *DestType.skipTypedefs()
      and (not SrcType.isScalar() or not DestType.isScalar())) {
    revng_log(Log,
              "WARNING: emitting a invalid bitcast in C, using "
              "__builtin_bit_cast");
    return (llvm::Twine("__builtin_bit_cast(") + B.getTypeName(DestType) + ", "
            + ExprToCast + ")")
      .str();
  }

  return addAlwaysParentheses(B.getTypeName(DestType)) + " "
         + addParentheses(ExprToCast);
}

static std::string getUndefToken(const model::Type &UndefType,
                                 const ptml::ModelCBuilder &B) {
  return "undef(" + B.getTypeName(UndefType) + ")";
}

static std::string getFormattedIntegerToken(const llvm::CallInst *Call,
                                            const ptml::ModelCBuilder &B) {

  if (isCallToTagged(Call, FunctionTags::HexInteger)) {
    const auto Operand = Call->getArgOperand(0);
    const auto *Value = cast<llvm::ConstantInt>(Operand);
    return B.getConstantTag(hexLiteral(Value, B)).toString();
  }

  if (isCallToTagged(Call, FunctionTags::CharInteger)) {
    const auto Operand = Call->getArgOperand(0);
    const auto *Value = cast<llvm::ConstantInt>(Operand);
    return B.getConstantTag(charLiteral(Value)).toString();
  }

  if (isCallToTagged(Call, FunctionTags::BoolInteger)) {
    const auto Operand = Call->getArgOperand(0);
    const auto *Value = cast<llvm::ConstantInt>(Operand);
    return B.getConstantTag(boolLiteral(Value)).toString();
  }

  if (isCallToTagged(Call, FunctionTags::NullPtr)) {
    const auto Operand = Call->getArgOperand(0);
    const auto *Value = cast<llvm::ConstantInt>(Operand);
    revng_assert(Value->isZero());
    return B.getNullTag().toString();
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
    rc_return getUndefToken(*TypeMap.at(Undef), B);

  if (auto *Null = dyn_cast<llvm::ConstantPointerNull>(C))
    rc_return B.getNullTag().toString();

  if (auto *Const = dyn_cast<llvm::ConstantInt>(C)) {
    llvm::APInt Value = Const->getValue();
    if (Value.isIntN(64))
      rc_return B.getNumber(Value).toString();
    else
      rc_return get128BitIntegerHexConstant(Value, B);
  }

  if (auto *Function = dyn_cast<llvm::Function>(C)) {
    const model::Function *ModelFunc = llvmToModelFunction(Model, *Function);
    revng_assert(ModelFunc);
    rc_return B.getReferenceTag(*ModelFunc);
  }

  if (auto *Global = dyn_cast<llvm::GlobalVariable>(C)) {
    using namespace llvm;
    // Check if initializer is a CString
    auto *Initializer = Global->getInitializer();

    StringRef Content;
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
      const model::Type &SrcType = *TypeMap.at(Operand);
      const model::Type &DstType = *TypeMap.at(ConstExpr);

      // IntToPtr has no effect on values that we already know to be pointers
      if (SrcType.isPointer())
        rc_return rc_recur getConstantToken(Operand);
      else
        rc_return buildCastExpr(rc_recur getConstantToken(Operand),
                                SrcType,
                                DstType);
    }

    case Instruction::PtrToInt: {
      const auto *Operand = cast<llvm::Constant>(ConstExpr->getOperand(0));
      const model::Type &SrcType = *TypeMap.at(Operand);
      const model::Type &DstType = *TypeMap.at(ConstExpr);
      rc_return buildCastExpr(rc_recur getConstantToken(Operand),
                              SrcType,
                              DstType);
    }

    default:
      revng_abort(dumpToString(ConstExpr).c_str());
    }
  }

  if (isCallToTagged(C, FunctionTags::LiteralPrintDecorator))
    rc_return getFormattedIntegerToken(cast<llvm::CallInst>(C), B);

  std::string Error = "Cannot get token for llvm::Constant: ";
  Error += dumpToString(C).c_str();
  revng_abort(Error.c_str());

  rc_return "";
}

RecursiveCoroutine<std::string>
CCodeGenerator::getModelGEPToken(const llvm::CallInst *Call) const {

  revng_assert(isCallToTagged(Call, FunctionTags::ModelGEP)
               or isCallToTagged(Call, FunctionTags::ModelGEPRef));

  revng_assert(Call->arg_size() >= 2);

  bool IsRef = isCallToTagged(Call, FunctionTags::ModelGEPRef);

  // First argument is a string containing the base type
  auto *CurArg = Call->arg_begin();
  model::UpcastableType CurType = fromLLVMString(CurArg->get(), Model);

  // Second argument is the base llvm::Value
  ++CurArg;
  llvm::Value *BaseValue = CurArg->get();
  std::string BaseString = rc_recur getToken(BaseValue);

  bool UseArrow = false;
  if (IsRef) {
    // In ModelGEPRefs, the base value is a reference, and the base type is
    // its type
    const model::Type &Base = *TypeMap.at(BaseValue)->skipTypedefs();
    const model::Type &Cur = *std::as_const(CurType)->skipTypedefs();
    if (Base != Cur) {
      BaseValue->dump();
      TypeMap.at(BaseValue)->dump();
      CurType->dump();
      revng_abort("The ModelGEP base type is not coherent with the "
                  "propagated type.");
    }
    // If there are no further arguments we're just dereferencing the base value
    if (std::next(CurArg) == Call->arg_end()) {
      // But dereferencing a reference does not produce any code so we're done
      rc_return BaseString;
    }
  } else {
    // In ModelGEPs, the base value is a pointer, and the base type is the
    // type pointed by the base value
    const model::Type &Pointee = TypeMap.at(BaseValue)->getPointee();
    revng_assert(Pointee == *CurType,
                 "The ModelGEP base type is not coherent with the propagated "
                 "type.");

    auto *ThirdArgument = Call->getArgOperand(2);
    auto *ConstantArrayIndex = dyn_cast<llvm::ConstantInt>(ThirdArgument);

    // Check if the ModelGEP represents an additional access with square
    // brackets on the pointer
    bool HasInitialArrayAccess = not ConstantArrayIndex
                                 or not ConstantArrayIndex->isZero();

    // If this doesn't have any variadic argument just dereference the base
    // pointer and we're done.
    if (Call->arg_size() < 4) {
      // There are actually various ways to do it.

      // If we're not using square brackets to dereference the pointer, we just
      // emit a dereference expression.
      if (not HasInitialArrayAccess)
        rc_return buildDerefExpr(BaseString);

      // Here we have square brackets, that effectively replace the dereference
      // operator, so we just emit the square brackets with the appropriate
      // index.
      std::string IndexExpr;
      if (auto *Const = dyn_cast<llvm::ConstantInt>(ThirdArgument)) {
        IndexExpr = B.getNumber(Const->getValue()).toString();
      } else {
        IndexExpr = rc_recur getToken(ThirdArgument);
      }

      rc_return BaseString + "[" + IndexExpr + "]";
    }

    // Here we know that there is at least one variadic argument.

    if (HasInitialArrayAccess) {
      // If we're using the square brackets to dereference the base pointer we
      // have to change the base type so that it represents the "fake" array
      // being accessed.
      // We make it with only 1 element because in the following the number of
      // elements of the array is not actually used for generating the C code,
      // so we can get away with it.
      CurType = model::ArrayType::make(std::move(CurType), 1);
    } else {
      // Otherwise, we're not accessing the base pointer as an array.
      // So we can skip an additional argument.
      ++CurArg;

      // But the base type could still be an array.
      if (CurType->isArray()) {
        // If the base type is an array the first level of indirection will be
        // represented by square brackets that want to access elements of the
        // array. So we have to first dereference the pointer-to-array in order
        // to be able to access elements via [] in C.
        BaseString = "(" + buildDerefExpr(BaseString) + ")";
      } else {
        // If CurType is not an array we're going to represent the first level
        // of the traversal with the `->` operator rather than `.`, so let's
        // take note of this fact.
        UseArrow = true;
      }
    }
  }
  ++CurArg;

  std::string CurExpr = addParentheses(BaseString);
  using PTMLOperator = ptml::CBuilder::Operator;
  Tag Deref = UseArrow ? B.getOperator(PTMLOperator::Arrow) :
                         B.getOperator(PTMLOperator::Dot);

  // Traverse the model to decide whether to emit "." or "[]"
  for (; CurArg != Call->arg_end(); ++CurArg) {
    const auto &Unwrapped = *std::as_const(CurType)->skipConstAndTypedefs();
    if (const auto *D = llvm::dyn_cast<model::DefinedType>(&Unwrapped)) {
      // If it's a struct or union, we can only navigate it with fixed
      // indexes.
      // TODO: decide how to emit constants
      auto *FieldIdxConst = cast<llvm::ConstantInt>(CurArg->get());
      uint64_t FieldIdx = FieldIdxConst->getValue().getLimitedValue();

      CurExpr += Deref.toString();

      // Find the field name
      const model::TypeDefinition &Definition = D->unwrap();
      if (auto *S = dyn_cast<model::StructDefinition>(&Definition)) {
        const model::StructField &Field = S->Fields().at(FieldIdx);
        CurExpr += B.getReferenceTag(*S, Field);
        CurType = std::move(S->Fields().at(FieldIdx).Type());

      } else if (auto *U = dyn_cast<model::UnionDefinition>(&Definition)) {
        const model::UnionField &Field = U->Fields().at(FieldIdx);
        CurExpr += B.getReferenceTag(*U, Field);
        CurType = std::move(U->Fields().at(FieldIdx).Type());

      } else {
        CurType->dump();
        revng_abort("Unexpected ModelGEP type found: ");
      }

    } else if (auto *Array = llvm::dyn_cast<model::ArrayType>(&Unwrapped)) {
      std::string IndexExpr;
      if (auto *Const = dyn_cast<llvm::ConstantInt>(CurArg->get())) {
        IndexExpr = B.getNumber(Const->getValue()).toString();
      } else {
        IndexExpr = rc_recur getToken(CurArg->get());
      }

      CurExpr += "[" + IndexExpr + "]";
      CurType = std::move(Array->ElementType());

    } else {
      revng_abort("Integers/pointers do not have fields.");
    }

    // Regardless if the base type was a pointer or not, we are now
    // navigating only references
    Deref = B.getOperator(PTMLOperator::Dot);
  }

  rc_return CurExpr;
}

RecursiveCoroutine<std::string>
CCodeGenerator::getCustomOpcodeToken(const llvm::CallInst *Call) const {

  if (isAssignment(Call)) {
    const llvm::Value *StoredVal = Call->getArgOperand(0);
    const llvm::Value *PointerVal = Call->getArgOperand(1);
    rc_return rc_recur getToken(PointerVal) + " "
      + B.getOperator(ptml::CBuilder::Operator::Assign) + " "
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
    model::UpcastableType CurType = fromLLVMString(CurArg->get(), Model);

    // Second argument is the base llvm::Value
    ++CurArg;
    llvm::Value *BaseValue = CurArg->get();
    std::string Token = rc_recur getToken(BaseValue);

    // If it is an implicit cast, omit it.
    bool IsImplicit = cast<llvm::ConstantInt>(Call->getArgOperand(2))->isOne();
    if (IsImplicit)
      rc_return Token;

    // Emit the parenthesized cast expr, and we are done.
    rc_return buildCastExpr(Token, *TypeMap.at(BaseValue), *CurType);
  }

  if (isCallToTagged(Call, FunctionTags::AddressOf)) {
    // First operand is the type of the value being addressed (should not
    // introduce casts)
    const model::UpcastableType ArgType = fromLLVMString(Call->getArgOperand(0),
                                                         Model);

    // Second argument is the value being addressed
    llvm::Value *Arg = Call->getArgOperand(1);
    revng_assert(*ArgType->skipTypedefs() == *TypeMap.at(Arg)->skipTypedefs());

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
    auto StrucTypeName = B.getFunctionReturnType(Prototype);
    std::string StructInit = addAlwaysParentheses(StrucTypeName);

    // Emit RHS
    llvm::StringRef Separator = " {";
    for (const auto &Arg : Call->args()) {
      StructInit += Separator.str() + " " + rc_recur getToken(Arg);
      Separator = ",";
    }
    StructInit += " }";

    rc_return StructInit;
  }

  if (isCallToTagged(Call, FunctionTags::OpaqueExtractValue)) {

    const llvm::Value *AggregateOp = Call->getArgOperand(0);
    revng_assert(isArtificialAggregateLocalVarDecl(AggregateOp)
                 or isHelperAggregateLocalVarDecl(AggregateOp));

    const auto *CallReturnsStruct = llvm::cast<llvm::CallInst>(AggregateOp);
    const auto *I = llvm::cast<llvm::ConstantInt>(Call->getArgOperand(1));
    uint64_t Index = I->getZExtValue();

    std::string StructFieldRef;
    if (isArtificialAggregateLocalVarDecl(AggregateOp)) {
      const auto *CalleePrototype = getCallSitePrototype(Model,
                                                         CallReturnsStruct);
      auto RF = llvm::cast<const model::RawFunctionDefinition>(CalleePrototype);
      const auto &ReturnValue = std::next(RF->ReturnValues().begin(), Index);
      StructFieldRef = B.NameBuilder.name(*RF, *ReturnValue);
    } else if (isHelperAggregateLocalVarDecl(AggregateOp)) {
      // The call returning a struct is a call to a helper function.
      // It must be a direct call.
      const llvm::Function *Callee = getCalledFunction(CallReturnsStruct);
      revng_assert(Callee);
      StructFieldRef = getReturnStructFieldReferenceTag(Callee, Index, B);
    }

    rc_return rc_recur getToken(AggregateOp) + "." + StructFieldRef;
  }

  if (isCallToTagged(Call, FunctionTags::SegmentRef)) {
    auto *Callee = getCalledFunction(Call);
    const auto &[StartAddress,
                 VirtualSize] = extractSegmentKeyFromMetadata(*Callee);
    model::Segment Segment = Model.Segments().at({ StartAddress, VirtualSize });
    rc_return B.getReferenceTag(Segment);
  }

  if (isCallToTagged(Call, FunctionTags::Copy))
    rc_return rc_recur getToken(Call->getArgOperand(0));

  if (isCallToTagged(Call, FunctionTags::OpaqueCSVValue)) {
    auto *Callee = getCalledFunction(Call);
    std::string HelperRef = getHelperFunctionReferenceTag(Callee, B);
    rc_return rc_recur getCallToken(Call, HelperRef, /*prototype=*/nullptr);
  }

  using PTMLOperator = ptml::CBuilder::Operator;
  if (isCallToTagged(Call, FunctionTags::UnaryMinus)) {
    auto Operand = Call->getArgOperand(0);
    std::string ToNegate = rc_recur getToken(Operand);
    rc_return B.getOperator(PTMLOperator::UnaryMinus) + ToNegate;
  }

  if (isCallToTagged(Call, FunctionTags::BinaryNot)) {
    auto Operand = Call->getArgOperand(0);
    std::string ToNegate = rc_recur getToken(Operand);
    rc_return(Operand->getType()->isIntegerTy(1) ?
                B.getOperator(PTMLOperator::BoolNot) :
                B.getOperator(PTMLOperator::BinaryNot))
      + ToNegate;
  }

  if (isCallToTagged(Call, FunctionTags::BooleanNot)) {
    auto Operand = Call->getArgOperand(0);
    std::string ToNegate = rc_recur getToken(Operand);
    rc_return B.getOperator(PTMLOperator::BoolNot) + ToNegate;
  }

  if (isCallToTagged(Call, FunctionTags::StringLiteral)) {
    const auto Operand = Call->getArgOperand(0);
    std::string StringLiteral = rc_recur getToken(Operand);
    rc_return B.getStringLiteral(StringLiteral).toString();
  }

  if (isCallToTagged(Call, FunctionTags::Comment)) {
    const auto *Index = llvm::cast<llvm::ConstantInt>(Call->getArgOperand(0));
    const auto &Comment = ModelFunction.Comments().at(Index->getZExtValue());

    std::string ShouldBeEmittedAtOp = rc_recur getToken(Call->getArgOperand(2));
    llvm::StringRef ShouldBeEmittedAt = llvm::StringRef(ShouldBeEmittedAtOp);
    ShouldBeEmittedAt.consume_front("\"");
    ShouldBeEmittedAt.consume_back("\"");

    std::string IsBeingEmittedAtOp = rc_recur getToken(Call->getArgOperand(3));
    llvm::StringRef IsBeingEmittedAt = llvm::StringRef(IsBeingEmittedAtOp);
    IsBeingEmittedAt.consume_front("\"");
    IsBeingEmittedAt.consume_back("\"");

    const auto *IsExact = llvm::cast<llvm::ConstantInt>(Call->getArgOperand(1));
    if (IsExact->getZExtValue())
      revng_assert(ShouldBeEmittedAt == IsBeingEmittedAt);

    rc_return "\n"
      + B.getStatementComment(Comment,
                              pipeline::locationString(ranks::StatementComment,
                                                       ModelFunction.key(),
                                                       Index->getZExtValue()),
                              IsBeingEmittedAt);
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
      CalleeToken = B.getReferenceTag(DynamicFunc);
    } else {
      // Isolated function
      llvm::Function *CalledFunc = getCalledFunction(Call);
      revng_assert(CalledFunc);
      const model::Function *ModelFunc = llvmToModelFunction(Model,
                                                             *CalledFunc);
      revng_assert(ModelFunc);
      CalleeToken = B.getReferenceTag(*ModelFunc);
    }
  }

  // Build the call expression
  revng_assert(not CalleeToken.empty());
  const auto *Prototype = getCallSitePrototype(Model, Call);
  rc_return rc_recur getCallToken(Call, CalleeToken, Prototype);
}

RecursiveCoroutine<std::string>
CCodeGenerator::getNonIsolatedCallToken(const llvm::CallInst *Call) const {
  auto *CalledFunc = getCalledFunction(Call);
  revng_assert(CalledFunc and CalledFunc->hasName(),
               "Special functions should all have a name");

  std::string HelperRef = getHelperFunctionReferenceTag(CalledFunc, B);
  rc_return rc_recur getCallToken(Call, HelperRef, /*prototype=*/nullptr);
}

static std::string addDebugInfo(const llvm::Instruction *I,
                                std::string &&Str,
                                const ptml::ModelCBuilder &B) {
  if (I->getDebugLoc() && I->getDebugLoc()->getScope()) {
    std::string Location = I->getDebugLoc()->getScope()->getName().str();
    return B.getDebugInfoTag(std::move(Str), std::move(Location));

  } else {
    return std::move(Str);
  }
}

/// Return the string that represents the given binary operator in C
static const std::string getBinOpString(const llvm::BinaryOperator *BinOp,
                                        const ptml::CBuilder &B) {
  const Tag Op = [&BinOp, &B]() {
    bool IsBool = BinOp->getType()->isIntegerTy(1);

    using PTMLOperator = ptml::CBuilder::Operator;

    switch (BinOp->getOpcode()) {
    case Instruction::Add:
      return B.getOperator(ptml::CBuilder::Operator::Add);
    case Instruction::Sub:
      return B.getOperator(ptml::CBuilder::Operator::Sub);
    case Instruction::Mul:
      return B.getOperator(ptml::CBuilder::Operator::Mul);
    case Instruction::SDiv:
    case Instruction::UDiv:
      return B.getOperator(ptml::CBuilder::Operator::Div);
    case Instruction::SRem:
    case Instruction::URem:
      return B.getOperator(ptml::CBuilder::Operator::Modulo);
    case Instruction::LShr:
    case Instruction::AShr:
      return B.getOperator(ptml::CBuilder::Operator::RShift);
    case Instruction::Shl:
      return B.getOperator(ptml::CBuilder::Operator::LShift);
    case Instruction::And:
      return IsBool ? B.getOperator(PTMLOperator::BoolAnd) :
                      B.getOperator(ptml::CBuilder::Operator::And);
    case Instruction::Or:
      return IsBool ? B.getOperator(PTMLOperator::BoolOr) :
                      B.getOperator(ptml::CBuilder::Operator::Or);
    case Instruction::Xor:
      return B.getOperator(ptml::CBuilder::Operator::Xor);
    default:
      revng_abort("Unknown const Binary operation");
    }
  }();
  return " " + Op + " ";
}

/// Return the string that represents the given comparison operator in C
static const std::string getCmpOpString(const llvm::CmpInst::Predicate &Pred,
                                        const ptml::CBuilder &B) {
  using llvm::CmpInst;
  const Tag Op = [&Pred, &B]() {
    switch (Pred) {
    case CmpInst::ICMP_EQ: ///< equal
      return B.getOperator(ptml::CBuilder::Operator::CmpEq);
    case CmpInst::ICMP_NE: ///< not equal
      return B.getOperator(ptml::CBuilder::Operator::CmpNeq);
    case CmpInst::ICMP_UGT: ///< unsigned greater than
    case CmpInst::ICMP_SGT: ///< signed greater than
      return B.getOperator(ptml::CBuilder::Operator::CmpGt);
    case CmpInst::ICMP_UGE: ///< unsigned greater or equal
    case CmpInst::ICMP_SGE: ///< signed greater or equal
      return B.getOperator(ptml::CBuilder::Operator::CmpGte);
    case CmpInst::ICMP_ULT: ///< unsigned less than
    case CmpInst::ICMP_SLT: ///< signed less than
      return B.getOperator(ptml::CBuilder::Operator::CmpLt);
    case CmpInst::ICMP_ULE: ///< unsigned less or equal
    case CmpInst::ICMP_SLE: ///< signed less or equal
      return B.getOperator(ptml::CBuilder::Operator::CmpLte);
    default:
      revng_abort("Unknown comparison operator");
    }
  }();
  return " " + Op + " ";
}

RecursiveCoroutine<std::string>
CCodeGenerator::getInstructionToken(const llvm::Instruction *I) const {

  if (isa<llvm::BinaryOperator>(I) or isa<llvm::ICmpInst>(I)) {
    const llvm::Value *Op0 = I->getOperand(0);
    const llvm::Value *Op1 = I->getOperand(1);

    std::string Op0Token = rc_recur getToken(Op0);
    std::string Op1Token = rc_recur getToken(Op1);

    const model::Type &OpType0 = *TypeMap.at(Op0);
    const model::Type &OpType1 = *TypeMap.at(Op1);

    revng_assert(OpType0.isScalar() and OpType1.isScalar());

    // In principle, OpType0 and OpType1 should always have the same size.
    // There is a notable exception though: we use LLVM with a 64bit DataLayout
    // for which pointers are always 64-bits wide on LLVM IR, while they can be
    // 32-bits wide on the model, depending on the binary we're decompiling.
    // So the only situation where sizes are allowed to mismatch is when one of
    // the operands is a pointer (on the model) and the other isn't.
    if (*OpType0.size() != *OpType1.size()) {
      // If this happens, only one of the two operands must be a pointer, and
      // the other must be a constant integer that fits in the pointer size, at
      // most masked behind a decorator.
      revng_assert(OpType0.isPointer() xor OpType1.isPointer());
      const model::Type &PointerModelType = OpType0.isPointer() ? OpType0 :
                                                                  OpType1;
      const model::Type &IntegerModelType = OpType0.isPointer() ? OpType1 :
                                                                  OpType0;
      auto PointerByteSize = *PointerModelType.size();
      auto IntegerByteSize = *IntegerModelType.size();
      revng_assert(PointerByteSize < IntegerByteSize);

      const llvm::Value *IntegerOperand = OpType0.isPointer() ? Op1 : Op0;
      auto *Integer = dyn_cast<llvm::ConstantInt>(IntegerOperand);
      if (not Integer) {
        using namespace FunctionTags;
        auto *CallToDecorator = getCallToTagged(IntegerOperand,
                                                LiteralPrintDecorator);
        revng_assert(CallToDecorator);
        Integer = cast<llvm::ConstantInt>(CallToDecorator->getArgOperand(0));
      }
      revng_assert(nullptr != Integer);
    }

    auto *Bin = dyn_cast<llvm::BinaryOperator>(I);
    auto *Cmp = dyn_cast<llvm::ICmpInst>(I);
    revng_assert(Bin or Cmp);
    auto OperatorString = Bin ? getBinOpString(Bin, B) :
                                getCmpOpString(Cmp->getPredicate(), B);

    // TODO: Integer promotion
    rc_return addDebugInfo(I,
                           addParentheses(Op0Token) + OperatorString
                             + addParentheses(Op1Token),
                           B);
  }

  if (isa<llvm::CastInst>(I) or isa<llvm::FreezeInst>(I)) {
    // Those are usually noops on the LLVM IR.
    const llvm::Value *Op = I->getOperand(0);
    rc_return addDebugInfo(I, rc_recur getToken(Op), B);
  }

  switch (I->getOpcode()) {

  case llvm::Instruction::Call: {
    auto *Call = cast<llvm::CallInst>(I);

    revng_assert(isCallToCustomOpcode(Call) or isCallToIsolatedFunction(Call)
                 or isCallToNonIsolated(Call));

    if (isCallToCustomOpcode(Call))
      rc_return addDebugInfo(I, rc_recur getCustomOpcodeToken(Call), B);

    if (isCallToIsolatedFunction(Call))
      rc_return addDebugInfo(I, rc_recur getIsolatedCallToken(Call), B);

    if (isCallToNonIsolated(Call))
      rc_return addDebugInfo(I, rc_recur getNonIsolatedCallToken(Call), B);

    std::string Error = "Cannot get token for CallInst: " + dumpToString(Call);
    revng_abort(Error.c_str());

    rc_return "";
  }

  case llvm::Instruction::Ret: {

    std::string Result = B.getKeyword(ptml::CBuilder::Keyword::Return)
                           .toString();
    if (auto *Ret = llvm::cast<llvm::ReturnInst>(I);
        llvm::Value *ReturnedVal = Ret->getReturnValue())
      Result += " " + rc_recur getToken(ReturnedVal);

    rc_return addDebugInfo(I, std::move(Result), B);
  }

  case llvm::Instruction::Unreachable:
    rc_return addDebugInfo(I, "__builtin_trap()", B);

  case llvm::Instruction::Select: {

    auto *Select = llvm::cast<llvm::SelectInst>(I);
    std::string Condition = rc_recur getToken(Select->getCondition());
    const llvm::Value *Op1 = Select->getOperand(1);
    const llvm::Value *Op2 = Select->getOperand(2);

    std::string Op1String = rc_recur getToken(Op1);
    std::string Op2String = rc_recur getToken(Op2);

    rc_return addDebugInfo(I,
                           addParentheses(Condition) + " ? "
                             + addParentheses(Op1String) + " : "
                             + addParentheses(Op2String),
                           B);
  }

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
                 or isCallStackArgumentDecl(V) or isLocalVarDecl(V)
                 or isArtificialAggregateLocalVarDecl(V)
                 or isHelperAggregateLocalVarDecl(V));
    revng_log(Log, "Found!");
    rc_return It->second;
  }

  // We should always have names for stuff that is expected to have a name.
  revng_assert(not isa<llvm::Argument>(V) and not isStackFrameDecl(V)
               and not isCallStackArgumentDecl(V) and not isLocalVarDecl(V)
               and not isArtificialAggregateLocalVarDecl(V)
               and not isHelperAggregateLocalVarDecl(V));

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
                             const model::TypeDefinition *Prototype) const {
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

void CCodeGenerator::emitBasicBlock(const llvm::BasicBlock *BB,
                                    bool EmitReturn) {
  LoggerIndent Indent{ VisitLog };
  revng_log(VisitLog, "|__ Visiting BB " << BB->getName());
  LoggerIndent MoreIndent{ VisitLog };
  revng_log(Log, "--------- BB " << BB->getName());

  bool NextStatementDoesNotReturn = false;

  for (const Instruction &I : *BB) {
    revng_log(Log, "Analyzing: " << dumpToString(I, ModuleSlotTracker));

    auto *Call = dyn_cast<llvm::CallInst>(&I);

    bool IsStatement = isStatement(I);

    if (not IsStatement) {
      revng_log(Log, "Ignoring: non-statement instruction");

    } else if (I.getType()->isVoidTy()) {
      revng_assert(isa<llvm::ReturnInst>(I) or isCallToIsolatedFunction(&I)
                   or isCallToNonIsolated(&I) or isAssignment(&I)
                   or isComment(&I));

      // Handle the implicit `return` emission. If the correct parameter is set,
      // avoid the emission of the `Instruction` token.
      if (not(llvm::isa<llvm::ReturnInst>(I) and not EmitReturn)) {
        B.append(std::string(getToken(&I)) + (not isComment(&I) ? ";\n" : ""));

        if (NextStatementDoesNotReturn)
          B.append("// The previous function call does not return\n");
      }

    } else if (isHelperAggregateLocalVarDecl(Call)
               or isArtificialAggregateLocalVarDecl(Call)) {
      // This is a call but it actually needs an assignment to the associated
      // variable. The variable has not been declared in the IR with
      // LocalVariable, because LocalVariable needs a model type, and aggregates
      // types on the LLVM IR are not on the model.
      revng_assert(Call->getType()->isAggregateType());

      std::string VarName = getVarName(Call);
      revng_assert(not VarName.empty());

      // Get the token. If the Call is a call to an isolated function that
      // returns an aggregate we want to get the token of the call, not of the
      // local variable. For all the other cases we can just get the regular
      // token.
      std::string RHSExpression = isArtificialAggregateLocalVarDecl(Call) ?
                                    getIsolatedCallToken(Call) :
                                    getNonIsolatedCallToken(Call);

      // Assign to the local variable
      B.append(VarName + " " + B.getOperator(ptml::CBuilder::Operator::Assign)
               + " " + std::move(RHSExpression) + ";\n");
    } else {
      std::string Error = "Cannot emit statement: ";
      Error += dumpToString(Call).c_str();
      revng_abort(Error.c_str());
    }

    if (IsStatement)
      NextStatementDoesNotReturn = false;

    if (Call != nullptr and isCallToIsolatedFunction(Call)) {
      const auto &[CallEdge, _] = Cache.getCallEdge(Model, Call);
      if (CallEdge->hasAttribute(Model, model::FunctionAttribute::NoReturn)) {
        NextStatementDoesNotReturn = true;
      }
    }
  }
}

RecursiveCoroutine<std::string>
CCodeGenerator::buildGHASTCondition(const ExprNode *E, bool EmitBB) {
  LoggerIndent Indent{ VisitLog };
  revng_log(VisitLog, "|__ Visiting Condition " << E);
  LoggerIndent MoreIndent{ VisitLog };

  using NodeKind = ExprNode::NodeKind;
  switch (E->getKind()) {

  case NodeKind::NK_ValueCompare:
  case NodeKind::NK_LoopStateCompare: {
    revng_log(VisitLog, "(compare)");

    // A compare node, is used to represent a pre-computed condition, which is
    // the result of a `switch` promotion to `if`. The compare node can appear
    // in multiple variants. Specifically, we may have that the LHS of the
    // condition is an actual `llvm::Value` on the IR, or it is a placeholder
    // for the loop state variable.
    const CompareNode *Compare = cast<CompareNode>(E);

    // String that will contain the serialization of the `CompareNode`
    std::string CompareNodeString;

    // Decide whether to emit the LHS in the form of a pre-existing
    // `llvm::Value` or the use of the `LoopStateVar`
    switch (E->getKind()) {
    case NodeKind::NK_ValueCompare: {
      revng_log(VisitLog, "(value compare)");
      const ValueCompareNode *ValueCompare = cast<ValueCompareNode>(E);

      // We emit the instruction in the basic block before the llvm::Value
      llvm::BasicBlock *BB = ValueCompare->getBasicBlock();
      revng_assert(BB != nullptr);

      // If we are emitting an `IfNode` which derives from the promotion of a
      // `DualSwitch`, which in turn was a weaved one, we should not double emit
      // the instructions that compute the condition, because they have been
      // already emitted by the above switch.
      if (EmitBB) {
        emitBasicBlock(BB, true);
      }

      // Retrieve the `llvm::Value` representing the switch condition
      llvm::Instruction *Terminator = BB->getTerminator();
      llvm::SwitchInst *SwitchInst = llvm::cast<llvm::SwitchInst>(Terminator);
      llvm::Value *ConditionValue = SwitchInst->getCondition();
      revng_assert(ConditionValue);

      // Emit the condition variable
      std::string ConditionVarString = getToken(ConditionValue);
      CompareNodeString += ConditionVarString;

    } break;
    case NodeKind::NK_LoopStateCompare: {
      revng_log(VisitLog, "(loop state compare)");

      // Insert the loop state variable representing string
      CompareNodeString += LoopStateVar;

    } break;
    default: {
      revng_abort();
    }
    }

    // If the `ComparisonKind` is of the `NotPresent` kind, we don't need to
    // print out the comparison operator nor the RHS
    auto Comparison = Compare->getComparison();
    if (Comparison != CompareNode::ComparisonKind::Comparison_NotPresent) {

      // We either generate the `==` or a `!=`, depending on the operator
      // contained in the `CompareNode`
      auto Comparison = Compare->getComparison();
      using Operator = ptml::CBuilder::Operator;
      switch (Comparison) {
      case CompareNode::ComparisonKind::Comparison_Equal: {
        auto CmpString = B.getOperator(Operator::CmpEq);
        CompareNodeString += " " + CmpString;
      } break;
      case CompareNode::ComparisonKind::Comparison_NotEqual: {
        auto CmpString = B.getOperator(Operator::CmpNeq);
        CompareNodeString += " " + CmpString;
      } break;
      default: {
        revng_abort();
      }
      }

      // Build the RHS comparison constant
      size_t Constant = Compare->getConstant();
      CompareNodeString += " " + B.getNumber(Constant);
    }

    rc_return CompareNodeString;
  }

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

    // If we are emitting an `IfNode` which derives from the promotion of a
    // `DualSwitch`, which in turn was a weaved one, we should not double emit
    // the instructions that compute the condition, because they have been
    // already emitted by the above switch.
    if (EmitBB) {
      emitBasicBlock(BB, true);
    }

    // Then, extract the token of the last instruction (must be a
    // conditional branch instruction)
    llvm::Instruction *CondTerminator = BB->getTerminator();
    llvm::BranchInst *Br = cast<llvm::BranchInst>(CondTerminator);
    revng_assert(Br->isConditional());

    // Emit code for x != 0 case with cast.
    auto *I = dyn_cast<llvm::Instruction>(Br->getCondition());
    if (I) {
      auto *Cmp = dyn_cast<llvm::CmpInst>(I);
      const llvm::Value *Op1 = Cmp != nullptr ? I->getOperand(1) : nullptr;
      if (Cmp and Cmp->getPredicate() == llvm::CmpInst::ICMP_NE
          and dyn_cast<llvm::Constant>(Op1)
          and cast<llvm::Constant>(Op1)->isZeroValue()) {

        const llvm::Value *Op0 = I->getOperand(0);
        rc_return addDebugInfo(I, rc_recur getToken(Op0), B);
      }
    }
    rc_return rc_recur getToken(Br->getCondition());
  }

  case NodeKind::NK_Not: {
    revng_log(VisitLog, "(not)");

    const NotNode *N = cast<NotNode>(E);
    ExprNode *Negated = N->getNegatedNode();
    rc_return B.getOperator(ptml::CBuilder::Operator::BoolNot)
      + addAlwaysParentheses(rc_recur buildGHASTCondition(Negated, EmitBB));
  }

  case NodeKind::NK_And:
  case NodeKind::NK_Or: {
    revng_log(VisitLog, "(and/or)");

    const BinaryNode *Binary = cast<BinaryNode>(E);

    const auto &[Child1, Child2] = Binary->getInternalNodes();
    std::string Child1Token = rc_recur buildGHASTCondition(Child1, EmitBB);
    std::string Child2Token = rc_recur buildGHASTCondition(Child2, EmitBB);
    using PTMLOperator = ptml::CBuilder::Operator;
    const Tag &OpToken = E->getKind() == NodeKind::NK_And ?
                           B.getOperator(PTMLOperator::BoolAnd) :
                           B.getOperator(PTMLOperator::BoolOr);
    rc_return addAlwaysParentheses(Child1Token) + " " + OpToken.toString() + " "
      + addAlwaysParentheses(Child2Token);
  }

  default:
    revng_abort("Unknown ExprNode kind");
  }
}

static std::string makeWhile(const ptml::CBuilder &B,
                             const std::string &CondExpr) {
  revng_assert(not CondExpr.empty());
  return B.getKeyword(ptml::CBuilder::Keyword::While).toString() + " ("
         + CondExpr + ")";
}

RecursiveCoroutine<void> CCodeGenerator::emitGHASTNode(const ASTNode *N) {
  if (N == nullptr)
    rc_return;

  auto VarToDeclareIt = VariablesToDeclare.find(N);
  if (VarToDeclareIt != VariablesToDeclare.end()) {
    for (const CallInst *VarDeclCall : VarToDeclareIt->second) {
      // Emit missing local variable declarations
      if (isLocalVarDecl(VarDeclCall) or isCallStackArgumentDecl(VarDeclCall)) {
        std::string VarName = createLocalVarDeclName(VarDeclCall);
        revng_assert(not VarName.empty());

        // TODO: drop this workaround when we properly support emission of
        // variable declarations with inline initialization.
        //
        // At the moment in C we can't emit variable declarations with inline
        // initialization. Not for some substantial problem, but we haven't
        // implemented it yet. As a result if TypeMap.at(VarDeclCall) returns a
        // const-qualified type we will end up generating C code that doesn't
        // compile, when it tries to assign a value to the variable for
        // initializing it separately from the declaration.
        // To work around this, until we don't support emission of variable
        // declarations with inline initialization, we have to strip away
        // constness.
        auto NonConst = model::getNonConst(*TypeMap.at(VarDeclCall));

        B.append(B.getNamedCInstance(*NonConst, VarName));
      } else if (isArtificialAggregateLocalVarDecl(VarDeclCall)) {
        // Create missing local variable declarations
        std::string VarName = createLocalVarDeclName(VarDeclCall);
        revng_assert(not VarName.empty());

        const auto *Prototype = getCallSitePrototype(Model, VarDeclCall);
        revng_assert(Prototype);
        B.append(B.getNamedCInstanceOfReturnType(*Prototype, VarName));

      } else if (isHelperAggregateLocalVarDecl(VarDeclCall)) {
        // Create missing local variable declarations
        std::string VarName = createLocalVarDeclName(VarDeclCall);
        revng_assert(not VarName.empty());
        llvm::Function *Callee = VarDeclCall->getCalledFunction();
        revng_assert(Callee);
        B.append(getReturnTypeReferenceTag(Callee, B) + " " + VarName);
      } else {
        revng_assert(not VarDeclCall->getType()->isAggregateType());
      }
      B.append(";\n");
    }
  }

  revng_log(VisitLog, "|__ GHAST Node " << N->getID());
  LoggerIndent Indent{ VisitLog };

  auto Kind = N->getKind();
  switch (Kind) {

  case ASTNode::NodeKind::NK_Break: {
    revng_log(VisitLog, "(NK_Break)");

    const BreakNode *Break = llvm::cast<BreakNode>(N);
    using PTMLOperator = ptml::CBuilder::Operator;
    if (Break->breaksFromWithinSwitch()) {
      revng_assert(not SwitchStateVars.empty()
                   and not SwitchStateVars.back().empty());
      B.append(SwitchStateVars.back() + " "
               + B.getOperator(PTMLOperator::Assign) + " " + B.getTrueTag()
               + ";\n");
    }
  }
    [[fallthrough]];

  case ASTNode::NodeKind::NK_SwitchBreak: {
    revng_log(VisitLog, "(NK_SwitchBreak)");

    B.append(B.getKeyword(ptml::CBuilder::Keyword::Break) + ";\n");
  } break;

  case ASTNode::NodeKind::NK_Continue: {
    revng_log(VisitLog, "(NK_Continue)");

    const ContinueNode *Continue = cast<ContinueNode>(N);

    // Print the condition computation code of the if statement.
    if (Continue->hasComputation()) {
      IfNode *ComputationIfNode = Continue->getComputationIfNode();
      bool EmitBB = not ComputationIfNode->isWeaved();
      rc_recur buildGHASTCondition(ComputationIfNode->getCondExpr(), EmitBB);
    }

    // Actually print the continue statement only if the continue is not
    // implicit (i.e. it is not the last statement of the loop).
    if (not Continue->isImplicit())
      B.append(B.getKeyword(ptml::CBuilder::Keyword::Continue) + ";\n");
  } break;

  case ASTNode::NodeKind::NK_Code: {
    revng_log(VisitLog, "(NK_Code)");

    const CodeNode *Code = cast<CodeNode>(N);
    llvm::BasicBlock *BB = Code->getOriginalBB();
    revng_assert(BB != nullptr);
    emitBasicBlock(BB, not Code->containsImplicitReturn());
  } break;

  case ASTNode::NodeKind::NK_If: {
    revng_log(VisitLog, "(NK_If)");

    const IfNode *If = cast<IfNode>(N);

    std::string CondExpr;
    if (If->getCondExpr()) {

      // If we are in presence of a standard `IfNode`, construct the `CondExpr`
      bool EmitBB = not If->isWeaved();
      CondExpr = rc_recur buildGHASTCondition(If->getCondExpr(), EmitBB);
    } else {

      // We are emitting a `IfNode` promoted from a dispatcher `SwitchNode` with
      // two `case`s
      CondExpr = LoopStateVar;
    }
    // "If" expression
    // TODO: possibly cast the CondExpr if it's not convertible to boolean?
    revng_assert(not CondExpr.empty());
    B.append(B.getKeyword(ptml::CBuilder::Keyword::If) + " (" + CondExpr
             + ") ");
    {
      Scope TheScope = B.getCurvedBracketScope();
      // "Then" expression (always emitted)
      if (nullptr == If->getThen())
        B.appendLineComment("Empty");
      else
        rc_recur emitGHASTNode(If->getThen());
    }

    // "Else" expression (optional)
    if (If->hasElse()) {
      B.append(" " + B.getKeyword(ptml::CBuilder::Keyword::Else) + " ");
      Scope TheScope = B.getCurvedBracketScope();
      rc_recur emitGHASTNode(If->getElse());
    }
    B.append("\n");
  } break;

  case ASTNode::NodeKind::NK_Scs: {
    revng_log(VisitLog, "(NK_Scs)");

    const ScsNode *Loop = cast<ScsNode>(N);

    std::string CondExpr;

    // Emit loop entry
    if (Loop->isDoWhile()) {
      B.append(B.getKeyword(ptml::CBuilder::Keyword::Do) + " ");
    } else {
      if (Loop->isWhileTrue()) {
        CondExpr = B.getTrueTag().toString();
      } else {
        revng_assert(Loop->isWhile());
        CondExpr = rc_recur makeLoopCondition(Loop->getRelatedCondition());
      }
      B.append(makeWhile(B, CondExpr) + " ");
    }

    {
      Scope TheScope = B.getCurvedBracketScope();
      if ((not Loop->hasBody()) and (not Loop->isDoWhile())) {
        revng_log(Log,
                  "WARNING: emitting a loop with an empty body which is not a "
                  "do-while");
      }
      if (Loop->hasBody())
        rc_recur emitGHASTNode(Loop->getBody());

      // If the loop is a do while we have to build the condition here, because
      // the computation of the condition must be emitted before TheScope is
      // closed to stay inside the loop body in C.
      if (Loop->isDoWhile()) {
        revng_assert(CondExpr.empty());
        CondExpr = rc_recur makeLoopCondition(Loop->getRelatedCondition());
      }
    }

    // Emit loop exit
    if (Loop->isDoWhile()) {
      B.append(" " + makeWhile(B, CondExpr) + ";");
    }
    B.append("\n");

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

      SortedVector<MetaAddress> Location;
      if (const llvm::BasicBlock *BasicBlock = Switch->getBB())
        if (not BasicBlock->empty())
          if (std::optional MA = revng::tryExtractAddress(BasicBlock->back()))
            Location.emplace(*MA);

      auto [Definition, Reference] = B.getVariableTags(VariableNameBuilder,
                                                       Location);
      SwitchStateVars.push_back(std::move(Reference));
      using COperator = ptml::CBuilder::Operator;
      B.append(B.tokenTag("bool", ptml::c::tokens::Type) + " "
               + std::move(Definition) + " " + B.getOperator(COperator::Assign)
               + " " + B.getFalseTag() + ";\n");
    }

    // Generate the condition of the switch
    StringToken SwitchVarToken;
    model::UpcastableType SwitchVarType;
    llvm::Value *SwitchVar = Switch->getCondition();
    if (SwitchVar) {
      // If the switch is not weaved we need to print the instructions in
      // the basic block before it.
      if (not Switch->isWeaved()) {
        llvm::BasicBlock *BB = Switch->getOriginalBB();
        revng_assert(BB != nullptr); // This is not a switch dispatcher.
        emitBasicBlock(BB, true);
      }
      std::string SwitchVarString = getToken(SwitchVar);
      SwitchVarToken = SwitchVarString;
      SwitchVarType = TypeMap.at(SwitchVar).copy();
    } else {
      revng_assert(Switch->getOriginalBB() == nullptr);
      revng_assert(!LoopStateVar.empty());
      // This switch does not come from an instruction: it's a dispatcher
      // for the loop state variable
      SwitchVarToken = LoopStateVar;

      // TODO: finer decision on the type of the loop state variable
      SwitchVarType = model::PrimitiveType::makeUnsigned(8);
    }
    revng_assert(not SwitchVarToken.empty());

    // Generate the switch statement
    B.append(B.getKeyword(ptml::CBuilder::Keyword::Switch).toString() + " ("
             + SwitchVarToken.str().str() + ") ");
    {
      Scope TheScope = B.getCurvedBracketScope();
      using PTMLKeyword = ptml::CBuilder::Keyword;

      // Generate the body of the switch (except for the default)
      for (const auto &[Labels, CaseNode] : Switch->cases_const_range()) {

        // If we encounter the `default` case, skip it, as it is emitted later
        if (Labels.empty() == true) {
          continue;
        }

        // Generate the case label(s) (multiple case labels might share the
        // same body)
        for (uint64_t CaseVal : Labels) {
          B.append(B.getKeyword(ptml::CBuilder::Keyword::Case) + " ");
          if (SwitchVar) {
            llvm::Type *SwitchVarT = SwitchVar->getType();
            auto *IntType = cast<llvm::IntegerType>(SwitchVarT);
            auto *CaseConst = llvm::ConstantInt::get(IntType, CaseVal);
            // TODO: assigned the signedness based on the signedness of the
            // condition
            B.append(B.getNumber(CaseConst->getValue()).toString());
          } else {
            B.append(B.getNumber(CaseVal).toString());
          }
          B.append(":\n");
        }

        {
          Scope InnerScope = B.getCurvedBracketScope();
          // Generate the case body
          rc_recur emitGHASTNode(CaseNode);
        }
        B.append(" " + B.getKeyword(PTMLKeyword::Break) + ";\n");
      }

      // Generate the default case if it exists
      if (auto *Default = Switch->getDefault()) {
        B.append(B.getKeyword(ptml::CBuilder::Keyword::Default) + ":\n");
        {
          Scope TheScope = B.getCurvedBracketScope();
          rc_recur emitGHASTNode(Default);
        }
        B.append(" " + B.getKeyword(PTMLKeyword::Break) + ";\n");
      }
    }
    B.append("\n");

    // If the switch needs a loop break dispatcher, reset the associated
    // state variable before emitting the switch statement.
    if (Switch->needsLoopBreakDispatcher()) {
      revng_assert(not SwitchStateVars.empty()
                   and not SwitchStateVars.back().empty());
      B.append(B.getKeyword(ptml::CBuilder::Keyword::If) + " ("
               + SwitchStateVars.back() + ")");
      {

        Scope ThenScope = B.getCurvedBracketScope();
        B.append(B.getKeyword(ptml::CBuilder::Keyword::Break) + ";");
      }
      B.append("\n");
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
    B.append(LoopStateVar + " "
             + B.getOperator(ptml::CBuilder::Operator::Assign) + " "
             + std::to_string(StateValue) + ";\n");
  } break;
  }

  rc_return;
}

static std::string getModelArgIdentifier(const model::TypeDefinition &ModelFT,
                                         const llvm::Argument &Argument,
                                         const ptml::ModelCBuilder &B) {
  const llvm::Function *LLVMFunction = Argument.getParent();
  unsigned ArgNo = Argument.getArgNo();

  if (auto *RFT = dyn_cast<model::RawFunctionDefinition>(&ModelFT)) {
    auto NumModelArguments = RFT->Arguments().size();
    revng_assert(ArgNo <= NumModelArguments + 1);
    revng_assert(LLVMFunction->arg_size() == NumModelArguments
                 or (not RFT->StackArgumentsType().isEmpty()
                     and (LLVMFunction->arg_size() == NumModelArguments + 1)));

    if (ArgNo < NumModelArguments) {
      const auto &Argument = *std::next(RFT->Arguments().begin(), ArgNo);
      return B.getReferenceTag(*RFT, Argument);

    } else {
      return B.getStackArgumentReferenceTag(*RFT);
    }

  } else if (auto *CFT = dyn_cast<model::CABIFunctionDefinition>(&ModelFT)) {
    revng_assert(LLVMFunction->arg_size() == CFT->Arguments().size());
    revng_assert(ArgNo < CFT->Arguments().size());
    return B.getReferenceTag(*CFT, CFT->Arguments().at(ArgNo));

  } else {
    revng_abort("Unexpected function type");
  }
}

void CCodeGenerator::emitFunction(bool NeedsLocalStateVar) {
  revng_log(Log, "========= Emitting Function " << LLVMFunction.getName());
  revng_log(VisitLog, "========= Function " << LLVMFunction.getName());
  LoggerIndent Indent{ VisitLog };

  auto FTagScope = B.getScopeTag(ptml::CBuilder::Scopes::Function);

  // Extract user comments from the model and emit them as PTML just before
  // the prototype.
  B.append(B.getFunctionComment(ModelFunction));

  // Print function's prototype
  B.printFunctionPrototype(Prototype, ModelFunction, false);

  // Set up the argument identifiers to be used in the function's body.
  for (const auto &Argument : LLVMFunction.args())
    TokenMap[&Argument] = getModelArgIdentifier(Prototype, Argument, B);

  // Print the function body
  B.append(" ");
  {
    Scope Body = B.getCurvedBracketScope(ptml::c::scopes::FunctionBody.str());

    // We expect just one stack type definition.
    bool IsStackDefined = false;

    // Declare the local variable representing the stack frame
    if (not ModelFunction.StackFrameType().isEmpty()) {
      revng_log(Log, "Stack Frame Declaration");
      const auto &IsStackFrameDecl = [](const llvm::Instruction &I) {
        return isStackFrameDecl(&I);
      };
      auto It = llvm::find_if(llvm::instructions(LLVMFunction),
                              IsStackFrameDecl);
      if (It != llvm::instructions(LLVMFunction).end()) {
        const auto *Call = &cast<llvm::CallInst>(*It);
        std::string VarName = createStackFrameVarDeclName(Call);
        revng_assert(not VarName.empty());

        revng_assert(not IsStackDefined, "Multiple stack variables?");
        const model::StructDefinition &Struct = *ModelFunction.stackFrameType();
        // In the artifacts generated by this LLVM-based backend we've given up
        // the requirement of emitting syntactically valid C code. Hence, we can
        // always emit the stack frame type definition inside the body of the
        // function itself, because we don't care anymore if it clashes with a
        // global definition that we have emitted in a header outside the
        // function's body.
        if (B.Configuration.EnableStackFrameInlining) {
          B.printDefinition(Struct, " " + std::move(VarName));
        } else {
          auto Named = B.getNamedCInstance(*ModelFunction.StackFrameType(),
                                           std::move(VarName));
          B.append(Named + ";\n");
        }
        IsStackDefined = true;

      } else {
        revng_log(Log,
                  "WARNING: function with valid stack type has no stack "
                  "declaration: "
                    << LLVMFunction.getName());
      }
    }

    // Emit a declaration for the loop state variable, which is used to
    // redirect control flow inside loops (e.g. if we want to jump in the
    // middle of a loop during a certain iteration)
    if (NeedsLocalStateVar)
      B.append(B.tokenTag("uint64_t", ptml::c::tokens::Type) + " "
               + LoopStateVarDeclaration + ";\n");

    // Recursively print the body of this function
    emitGHASTNode(GHAST.getRoot());

    // Mention unused variables by name, to reduce confusion when they suddenly
    // disappear.
    std::set<std::string> Homeless = VariableNameBuilder.homelessNames();
    if (not Homeless.empty()) {
      constexpr llvm::StringRef Separator = ", ";
      std::string Message = "The following variables are no longer present "
                            "in this function: ";
      for (llvm::StringRef Name : Homeless)
        Message += '`' + Name.str() + '`' + Separator.str();
      Message.resize(Message.size() - Separator.size());

      // TODO: embed an action into these variable definitions to allow
      //       dropping them easily.

      B.append('\n' + B.getFreeFormComment(std::move(Message)));
    }
  }

  B.append("\n");
}

static std::string decompileFunction(ControlFlowGraphCache &Cache,
                                     const llvm::Function &LLVMFunc,
                                     const ASTTree &CombedAST,
                                     const Binary &Model,
                                     const ASTVarDeclMap &VarToDeclare,
                                     bool NeedsLocalStateVar,
                                     ptml::ModelCBuilder &B) {
  std::string Result;

  llvm::raw_string_ostream Out(Result);
  B.setOutputStream(Out);

  CCodeGenerator Backend(Cache, Model, LLVMFunc, CombedAST, VarToDeclare, B);
  Backend.emitFunction(NeedsLocalStateVar);
  Out.flush();

  return Result;
}

static bool hasLoopDispatchers(const ASTTree &GHAST) {
  return needsLoopVar(GHAST.getRoot());
}

static ASTVarDeclMap computeVariableDeclarationScope(const llvm::Function &F,
                                                     const ASTTree &GHAST) {
  PendingVariableListType PendingVariables;
  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {

      auto *Call = dyn_cast<llvm::CallInst>(&I);
      if (not Call)
        continue;

      // Ignore the stack frame, which is handled separately.
      if (isStackFrameDecl(Call))
        continue;

      // All local variable declarations should go in the entry scope for now
      if (isLocalVarDecl(Call) or isCallStackArgumentDecl(Call)
          or isArtificialAggregateLocalVarDecl(Call)
          or isHelperAggregateLocalVarDecl(Call)) {
        PendingVariables.push_back(Call);
      }

      revng_assert(not isCallToNonIsolated(Call)
                   or not getCalledFunction(Call)->isTargetIntrinsic());
    }
  }

  return computeVarDeclMap(GHAST, PendingVariables);
}

/// Helper function which blanks a `llvm::Function` whose backend decompilation
/// failed, replaces its body with a single `BasicBlock` containing an error
/// message
static void turnBodyIntoError(llvm::Function &F) {
  using namespace llvm;

  // Save the previous linkage of `F`, so that it can be restored after
  // `deleteBody` is invoked
  auto FLinkage = F.getLinkage();

  // Save the `Metadata` attached to `F`
  llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>> SavedMetadata;
  F.getAllMetadata(SavedMetadata);

  // Remove all the content form `F`
  F.deleteBody();

  // Restore the linkage for `F`
  F.setLinkage(FLinkage);

  // Restore the `SavedMetadata`
  for (const auto &Pair : SavedMetadata) {
    F.setMetadata(Pair.first, Pair.second);
  }

  // Create the new empty error `BasicBlock`
  LLVMContext &Context = F.getContext();
  BasicBlock *NewBB = llvm::BasicBlock::Create(Context, "error", &F);

  // Add the error message
  revng::IRBuilder Builder(NewBB);
  emitAbort(Builder, "Backend Decompilation Failed");
}

/// Helper function which takes care of preparing the GHAST to emit the soft
/// fail message
static void softFail(llvm::Function &F, ASTTree &GHAST) {
  // Due to how the backend works, it is necessary to blank the body of the
  // `llvm::Function`, substitute it with a single `BasicBlock` containing the
  // error message, and call again the backend in order to decompile this new
  // version of the `Function`.

  // Blank the `GHAST` populated by the first run of `restructureCFG`
  GHAST = ASTTree();

  // Blank the function, and leave a single `BasicBlock` containing an error
  // message
  turnBodyIntoError(F);

  // Call again the `restructureCFG` pass on the new `Function` composed
  // only by the single `BasicBlock` with an error message
  bool NewRun = restructureCFG(F, GHAST);

  // The new run of `restructureCFG` must not fail
  revng_assert(NewRun);
}

std::string decompile(ControlFlowGraphCache &Cache,
                      llvm::Function &F,
                      const model::Binary &Model,
                      ptml::ModelCBuilder &B) {
  using namespace llvm;
  Task T2(3, Twine("decompile Function: ") + Twine(F.getName()));

  // TODO: this will eventually become a GHASTContainer for revng pipeline
  ASTTree GHAST;

  // Generate the GHAST and beautify it.
  {
    T2.advance("restructureCFG");

    // If `restructureCFG` failed, we want to provide as the decompiled output
    // a `Function` with an empty body containing an error message.
    if (not restructureCFG(F, GHAST)) {
      softFail(F, GHAST);
    }

    // TODO: beautification should be optional, but at the moment it's not
    // truly so (if disabled, things crash). We should strive to make it
    // optional for real.
    T2.advance("beautifyAST");

    // If `beautifyAST` failed, we want to provide as the decompiled output
    // a `Function` with an empty body containing an error message.
    if (not beautifyAST(Model, F, GHAST)) {
      softFail(F, GHAST);
    }
  }

  T2.advance("decompileFunction");
  if (Log.isEnabled()) {
    GHAST.dumpASTOnFile(F.getName().str(),
                        "ast-backend",
                        "AST-during-c-codegen.dot");
  }

  // Generated C code for F
  auto VariablesToDeclare = computeVariableDeclarationScope(F, GHAST);
  auto NeedsLoopStateVar = hasLoopDispatchers(GHAST);
  return decompileFunction(Cache,
                           F,
                           GHAST,
                           Model,
                           VariablesToDeclare,
                           NeedsLoopStateVar,
                           B);
}
