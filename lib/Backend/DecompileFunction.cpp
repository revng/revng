//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
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
#include "revng/Model/PrimitiveTypeKind.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Model/Segment.h"
#include "revng/Model/StructType.h"
#include "revng/Model/Type.h"
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

static constexpr const char *StackFrameVarName = "stack";

static Logger<> Log{ "c-backend" };
static Logger<> VisitLog{ "c-backend-visit-order" };
static Logger<> InlineLog{ "c-backend-inline" };

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

  // We always want to put the stack frame among the top-scope variables,
  // since it is is logical for it to appear at the top of the function even
  // if it is used only in a later scope.
  {
    bool Found = false;
    for (const BasicBlock &BB : F) {
      for (const Instruction &I : BB) {
        if (isCallTo(&I, "revng_stack_frame")) {
          revng_assert(not Found);
          TopScopeVars.insert(&I);
          Found = true;
        }
      }
    }
  }

  for (const BasicBlock &BB : F) {
    for (const Instruction &I : BB) {
      // All the others have already been promoted to LocalVariable Copy and
      // Assign.
      if (isCallToTagged(&I, FunctionTags::QEMU)
          or isCallToTagged(&I, FunctionTags::Helper)
          or isCallToTagged(&I, FunctionTags::Exceptional)
          or llvm::isa<llvm::IntrinsicInst>(I)) {

        const auto *Called = llvm::cast<CallInst>(&I)->getCalledFunction();
        revng_assert(not Called->isTargetIntrinsic());

        if (needsTopScopeDeclaration(I))
          TopScopeVars.insert(&I);
      }
    }
  }

  return TopScopeVars;
}

/// Helper function that also writes the logged string as a comment in the C
/// file if the corresponding logger is enabled
static void decompilerLog(llvm::raw_ostream &Out, const llvm::Twine &Expr) {
  revng_log(Log, Expr.str());
  if (InlineLog.isEnabled()) {
    auto CommentScope = helpers::BlockComment(Out);
    Out << Expr;
  }
}

static void debug_function dumpTokenMap(const TokenMapT &TokenMap) {
  llvm::dbgs() << "========== TokenMap ===========\n";
  for (auto [Value, Token] : TokenMap) {
    llvm::dbgs() << "Value: " << dumpToString(Value) << "\n";
    llvm::dbgs() << "Token: " << Token << "\n";
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "==============================\n";
}

static void debug_function dumpToString(llvm::Value *Value) {
  llvm::dbgs() << "Value: " << dumpToString(*Value) << "\n";
}

static StringToken addAlwaysParentheses(const llvm::Twine &Expr) {
  return StringToken(("(" + Expr + ")").str());
}

static StringToken get128BitIntegerHexConstant(llvm::APInt Value) {
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

  auto HighConstant = constants::constant(HighBitsString) + " "
                      + operators::LShift + " " + constants::number(64);
  auto CompositeConstant = HighConstant + " " + operators::Or + " "
                           + constants::constant(LowBitsString);
  return addAlwaysParentheses(CompositeConstant).str();
}

static StringToken hexLiteral(const llvm::ConstantInt *Int) {
  StringToken Formatted;
  if (Int->getBitWidth() <= 64) {
    Int->getValue().toString(Formatted,
                             /*radix*/ 16,
                             /*signed*/ false,
                             /*formatAsCLiteral*/ true);
    return Formatted;
  }
  return get128BitIntegerHexConstant(Int->getValue());
}

static StringToken charLiteral(const llvm::ConstantInt *Int) {
  revng_assert(Int->getValue().getBitWidth() == 8);
  const auto LimitedValue = Int->getLimitedValue(0xffu);
  const auto CharValue = static_cast<char>(LimitedValue);

  std::string EscapedC;
  llvm::raw_string_ostream EscapeCStream(EscapedC);
  EscapeCStream.write_escaped(std::string(&CharValue, 1));

  std::string EscapedHTML;
  llvm::raw_string_ostream EscapeHTMLStream(EscapedHTML);
  llvm::printHTMLEscaped(EscapedC, EscapeHTMLStream);

  return StringToken(llvm::formatv("'{0}'", EscapeHTMLStream.str()));
}

static StringToken boolLiteral(const llvm::ConstantInt *Int) {
  if (Int->isZero()) {
    return StringToken("false");
  } else {
    return StringToken("true");
  }
}

/// Return the string that represents the given binary operator in C
static const std::string getBinOpString(const llvm::BinaryOperator *BinOp) {
  const Tag *Op = [&BinOp]() constexpr {
    switch (BinOp->getOpcode()) {
    case Instruction::Add:
      return &operators::Add;
    case Instruction::Sub:
      return &operators::Sub;
    case Instruction::Mul:
      return &operators::Mul;
    case Instruction::SDiv:
    case Instruction::UDiv:
      return &operators::Div;
    case Instruction::SRem:
    case Instruction::URem:
      return &operators::Modulo;
    case Instruction::LShr:
    case Instruction::AShr:
      return &operators::RShift;
    case Instruction::Shl:
      return &operators::LShift;
    case Instruction::And:
      return &operators::And;
    case Instruction::Or:
      return &operators::Or;
    case Instruction::Xor:
      return &operators::Xor;
    default:
      revng_abort("Unknown const Binary operation");
    }
  }
  ();
  return " " + *Op + " ";
}

/// Return the string that represents the given comparison operator in C
static const std::string getCmpOpString(const llvm::CmpInst::Predicate &Pred) {
  using llvm::CmpInst;
  const Tag *Op = [&Pred]() constexpr {
    switch (Pred) {
    case CmpInst::ICMP_EQ: ///< equal
      return &operators::CmpEq;
    case CmpInst::ICMP_NE: ///< not equal
      return &operators::CmpNeq;
    case CmpInst::ICMP_UGT: ///< unsigned greater than
    case CmpInst::ICMP_SGT: ///< signed greater than
      return &operators::CmpGt;
    case CmpInst::ICMP_UGE: ///< unsigned greater or equal
    case CmpInst::ICMP_SGE: ///< signed greater or equal
      return &operators::CmpGte;
    case CmpInst::ICMP_ULT: ///< unsigned less than
    case CmpInst::ICMP_SLT: ///< signed less than
      return &operators::CmpLt;
    case CmpInst::ICMP_ULE: ///< unsigned less or equal
    case CmpInst::ICMP_SLE: ///< signed less or equal
      return &operators::CmpLte;
    default:
      revng_abort("Unknown comparison operator");
    }
  }
  ();
  return " " + *Op + " ";
}

/// Stateful name assignment for local variables.
class VarNameGenerator {
private:
  uint64_t CurVarID = 0;

public:
  StringToken nextVarName() {
    StringToken VarName("var_");
    VarName += to_string(CurVarID++);
    return VarName;
  }

  StringToken nextSwitchStateVar() {
    StringToken StateVar("break_from_loop_");
    StateVar += to_string(CurVarID++);
    return StateVar;
  }

  StringToken nextStackArgsVar() {
    StringToken StackArgsVar("stack_args_");
    StackArgsVar += to_string(CurVarID++);
    return StackArgsVar;
  }

  StringToken nextLocalStructName() {
    StringToken LocalStructName("local_struct_");
    LocalStructName += to_string(CurVarID++);
    return LocalStructName;
  }
};

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

  /// Name of the local variable used to break out of loops from within nested
  /// switches
  std::vector<std::string> SwitchStateVars;

  FunctionMetadataCache &Cache;

private:
  /// Stateful generator for variable names
  VarNameGenerator NameGenerator;
  /// Keep track of which expression is associated to each LLVM value during the
  /// emission of C code, specifically the PTML string representing:
  /// * In the case of a variable its use, which will have a location reference
  ///   (Indicating that the value of the variable has been used in that
  ///   instance)
  // clang-format off
  ///   \code{.c}
  ///   int foo = bar;
  ///           //^^^-- marked up with a 'location-reference' for variable bar
  ///   \endcode
  // clang-format on
  /// * In the case of an expression the combination of variables from above
  /// and,
  ///   when needed, the marked-up operators
  // clang-format off
  ///   \code{.c}
  ///   int baz = foo + bar;
  ///           //    ^------ operator '+' marked as such
  ///           //^^^---^^^-- both marked up with the respective variable
  ///   \endcode
  // clang-format on
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
                 raw_ostream &Out) :
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
    SwitchStateVars(),
    Cache(Cache) {
    // TODO: don't use a global loop state variable
    LoopStateVar = getVariableLocationReference("loop_state_var",
                                                ModelFunction);
    LoopStateVarDeclaration = getVariableLocationDefinition("loop_state_var",
                                                            ModelFunction);

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
  /// Whenever an atomic node is encountered, the associated basic block is
  /// emitted on-the-fly.
  RecursiveCoroutine<StringToken> buildGHASTCondition(const ExprNode *E);

  /// Serialize a basic block into a series of C statements.
  void emitBasicBlock(const BasicBlock *BB);

private:
  /// Emit an assignment for an instruction, if it is marked for assignment,
  /// otherwise add a string containing its expression to the TokenMap.
  /// Control-flow instructions are associated to their condition's token, and
  /// are never emitted directly (they are handled during the GHAST visit)
  StringToken buildExpression(const llvm::Instruction &I);

  /// Assign a string token and a QualifiedType to an Instruction
  /// operand. If the operand's value is an Instruction itself, the
  /// corresponding token must already exist.
  /// \return true if a new token has been generated for the operand
  RecursiveCoroutine<bool> addOperandToken(const llvm::Value *Operand);

  /// Emit a C-style function call. If the Prototype is null, it will be
  /// reconstructed from the LLVM types.
  StringToken buildFuncCallExpr(const llvm::CallInst *Call,
                                const llvm::StringRef FuncName,
                                const model::Type *Prototype = nullptr);

  /// Handle calls to functions that have been artificially injected by
  /// the decompiler pipeline and have a special meaning for decompilation.
  StringToken handleSpecialFunction(const llvm::CallInst *Call);

private:
  StringToken addParentheses(const llvm::Twine &Expr);

  StringToken buildDerefExpr(const llvm::Twine &Expr);

  StringToken buildAddressExpr(const llvm::Twine &Expr);

  /// Return a C string that represents a cast of \a ExprToCast to a given
  /// \a DestType. If no casting is needed between the two expression, the
  /// original expression is returned.
  StringToken buildCastExpr(StringRef ExprToCast,
                            const model::QualifiedType &SrcType,
                            const model::QualifiedType &DestType);

  /// Return a string that represents a C assignment:
  ///
  ///         [<TYPE>] LHSToken = [<CAST>] RHSToken
  ///
  /// \note <TYPE> is added if \a LHSTokens.Declaration is not empty (transforms
  /// the assignment into a declaration + assignment string)
  StringToken buildAssignmentExpr(const model::QualifiedType &LHSType,
                                  const VariableTokens &LHSTokens,
                                  const llvm::StringRef &RHSToken);

private:
  /// Implements special naming policies for special variables (e.g. stack
  /// frame)
  VariableTokens createVarName(const llvm::Value *V) {
    StringToken VarName;
    if (isCallTo(V, "revng_stack_frame")) {
      VarName = StackFrameVarName;
    } else if (isCallTo(V, "revng_call_stack_argument")) {
      VarName = NameGenerator.nextStackArgsVar();
    } else {
      VarName = NameGenerator.nextVarName();
    }
    return { getVariableLocationDefinition(VarName.str(), ModelFunction),
             getVariableLocationReference(VarName.str(), ModelFunction) };
  }

  /// Returns a variable name and a boolean indicating if the variable is new
  /// (i.e. it has never been declared) or it was already declared.
  VariableTokens getOrCreateVarName(const llvm::Value *V) {
    if (const auto *I = dyn_cast<Instruction>(V);
        I && TopScopeVariables.contains(I))
      return { TokenMap.at(I) };

    VariableTokens NewVar = createVarName(V);
    TokenMap[V] = NewVar.Use.str().str();
    return NewVar;
  }

private:
  /// Declare a local variable representing the given `Alloca` and return a
  /// token that represents is address.
  VariableTokens
  declareAllocaVariable(const llvm::AllocaInst *Alloca, VariableTokens Var) {
    // In LLVM IR, an alloca instruction returns a pointer, so the model type
    // associated to this value is actually a pointer to the model type of
    // the variable being allocated. Hence, to get the actual type of the
    // allocated variable, we must drop the pointer qualifier.
    const auto &AllocaType = TypeMap.at(Alloca);
    revng_assert(AllocaType.isPointer());
    QualifiedType AllocatedType = dropPointer(AllocaType);

    // Declare the variable
    Out << getNamedCInstance(AllocatedType, Var.Declaration) << ";\n";

    // Use the address of this variable as the token associated to the alloca
    return { Var.Declaration, operators::AddressOf + Var.Use };
  }
};

StringToken CCodeGenerator::addParentheses(const llvm::Twine &Expr) {
  if (IsOperatorPrecedenceResolutionPassEnabled)
    return StringToken(Expr.str());
  return StringToken(("(" + Expr + ")").str());
}

StringToken CCodeGenerator::buildDerefExpr(const llvm::Twine &Expr) {
  return StringToken(operators::PointerDereference + addParentheses(Expr));
}

StringToken CCodeGenerator::buildAddressExpr(const llvm::Twine &Expr) {
  return StringToken(operators::AddressOf + addParentheses(Expr));
}

StringToken
CCodeGenerator::buildCastExpr(StringRef ExprToCast,
                              const model::QualifiedType &SrcType,
                              const model::QualifiedType &DestType) {
  StringToken Result = ExprToCast;
  if (SrcType == DestType or not SrcType.UnqualifiedType().isValid()
      or not DestType.UnqualifiedType().isValid())
    return Result;

  revng_assert((SrcType.isScalar() or SrcType.isPointer())
               and (DestType.isScalar() or DestType.isPointer()));

  Result.assign(addAlwaysParentheses(getTypeName(DestType)));
  Result.append({ " ", addParentheses(ExprToCast) });

  return Result;
}

StringToken
CCodeGenerator::buildAssignmentExpr(const model::QualifiedType &LHSType,
                                    const VariableTokens &LHSTokens,
                                    const llvm::StringRef &RHSToken) {
  StringToken AssignmentStr;

  if (LHSTokens.hasDeclaration())
    AssignmentStr += getNamedCInstance(LHSType, LHSTokens.Declaration);
  else
    AssignmentStr += LHSTokens.Use;

  AssignmentStr += " " + operators::Assign + " ";
  AssignmentStr += RHSToken;

  return AssignmentStr;
}

StringToken
CCodeGenerator::buildFuncCallExpr(const llvm::CallInst *Call,
                                  const llvm::StringRef FuncName,
                                  const model::Type *Prototype /*=nullptr*/) {
  StringToken Expression;
  Expression += FuncName;

  if (Call->getNumArgOperands() == 0) {
    Expression += "()";

  } else {
    llvm::StringRef Separator = "(";

    for (const auto &Arg : Call->arg_operands()) {
      Expression += Separator;
      Expression += TokenMap.at(Arg);

      Separator = ", ";
    }
    Expression += ")";
  }

  return Expression;
}

RecursiveCoroutine<bool>
CCodeGenerator::addOperandToken(const llvm::Value *Operand) {
  revng_log(Log, "\tOperand: " << dumpToString(*Operand));

  if (Operand->getType()->isAggregateType()) {
    // Aggregate operands can either be:
    // 1. constant or global aggregates, which are not currently handled
    revng_assert(isa<llvm::Instruction>(Operand),
                 "The only aggregate operands should be generated by "
                 "special-cased instructions");
    // 2. aggregates returned by special-cased call functions, which should
    // have already added a token for this operand at this point
    revng_assert(TokenMap.contains(Operand),
                 "Tokens for aggregate operands should have already be "
                 "inserted when visiting the instruction tha returns such "
                 "aggregate.");
    rc_return false;
  }

  // Instructions must be visited in reverse-postorder when filling the
  // TokenMap
  if (isa<llvm::Instruction>(Operand) or isa<llvm::Argument>(Operand)) {
    if (auto *CallToCopy = isCallToTagged(Operand, FunctionTags::Copy)) {
      if (auto *LocalVar = isCallToTagged(CallToCopy->getArgOperand(0),
                                          FunctionTags::LocalVariable)) {
        revng_assert(TokenMap.contains(LocalVar),
                     dumpToString(LocalVar).c_str());
        TokenMap[Operand] = TokenMap.at(LocalVar);
        rc_return true;
      }
    }
    revng_assert(TokenMap.contains(Operand), dumpToString(Operand).c_str());
    rc_return false;
  }

  revng_assert(not Operand->getType()->isVoidTy());
  if (auto *Undef = dyn_cast<llvm::UndefValue>(Operand)) {
    revng_assert(Undef->getType()->isIntOrPtrTy());
    TokenMap[Operand] = constants::Zero.serialize();

    // Unfortunately, since the InlineLogger prints out values inside "/*"
    // comments, printing this if we're using such logger would break the C
    // code, since it would introduce unterminated "/*" comments.
    if (not InlineLog.isEnabled())
      TokenMap[Operand] += " " + helpers::blockComment("undef", false);
    rc_return true;
  }

  if (auto *Poison = dyn_cast<llvm::PoisonValue>(Operand)) {
    revng_assert(Poison->getType()->isIntOrPtrTy());
    TokenMap[Operand] = constants::Zero.serialize();

    // Unfortunately, since the InlineLogger prints out values inside "/*"
    // comments, printing this if we're using such logger would break the C
    // code, since it would introduce unterminated "/*" comments.
    if (not InlineLog.isEnabled())
      TokenMap[Operand] += " " + helpers::blockComment("poison", false);
    rc_return true;
  }

  if (auto *Null = dyn_cast<llvm::ConstantPointerNull>(Operand)) {
    TokenMap[Operand] = constants::Null.serialize();
    rc_return true;
  }

  if (auto *Glob = dyn_cast<llvm::GlobalVariable>(Operand)) {
    StringRef Name = Glob->getName();
    if (Name.size() > 0) {
      TokenMap[Operand] = Name;
      rc_return true;
    } else {
      rc_return false;
    }
  }

  if (auto *Const = dyn_cast<llvm::ConstantInt>(Operand)) {
    llvm::APInt Value = Const->getValue();
    if (Value.isIntN(64)) {
      // TODO: Decide how to print constants
      TokenMap[Operand] = constants::number(Value.getLimitedValue())
                            .serialize();
    } else {
      TokenMap[Operand] = get128BitIntegerHexConstant(Value).str().str();
    }

    rc_return true;
  }

  if (auto *ConstExpr = dyn_cast<llvm::ConstantExpr>(Operand)) {
    // A constant expression might have its own uninitialized constant
    // operands
    for (const llvm::Value *Op : ConstExpr->operand_values())
      rc_recur addOperandToken(Op);

    switch (ConstExpr->getOpcode()) {
    case Instruction::GetElementPtr: {
      using namespace llvm;
      auto *ResultType = ConstExpr->getType();
      auto *BasePointer = dyn_cast<GlobalVariable>(ConstExpr->getOperand(0));

      auto IsZero = [](llvm::Value *V) {
        if (auto *CI = dyn_cast<ConstantInt>(V))
          return CI->isZero();
        return false;
      };

      if (ResultType->isPointerTy()
          and ResultType->getPointerElementType()->isIntegerTy(8)
          and BasePointer != nullptr and BasePointer->isConstant()
          and BasePointer->hasInitializer() and ConstExpr->getNumOperands() == 3
          and IsZero(ConstExpr->getOperand(1))
          and IsZero(ConstExpr->getOperand(2))) {
        // Check if initializer is a CString
        auto *Initializer = BasePointer->getInitializer();
        auto *InitData = dyn_cast<ConstantDataSequential>(Initializer);
        if (InitData != nullptr and InitData->isCString()) {
          std::string Escaped;
          {
            raw_string_ostream Stream(Escaped);
            Stream << "\"";
            Stream.write_escaped(InitData->getAsString().drop_back());
            Stream << "\"";
          }
          TokenMap[ConstExpr] = std::move(Escaped);
          rc_return true;
        }
      }
    } break;
    case Instruction::IntToPtr: {
      const llvm::Value *ConstExprOperand = ConstExpr->getOperand(0);
      revng_assert(ConstExprOperand);

      const QualifiedType &SrcType = TypeMap.at(ConstExprOperand);
      const QualifiedType &DstType = TypeMap.at(ConstExpr);

      // IntToPtr has no effect on values that we already know to be pointers
      if (SrcType.isPointer())
        TokenMap[ConstExpr] = TokenMap.at(ConstExprOperand);
      else
        TokenMap[ConstExpr] = buildCastExpr(TokenMap.at(ConstExprOperand),
                                            SrcType,
                                            DstType)
                                .str()
                                .str();
      rc_return true;
    } break;
    }
  }

  rc_return false;
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

StringToken CCodeGenerator::handleSpecialFunction(const llvm::CallInst *Call) {

  auto *CalledFunc = Call->getCalledFunction();
  revng_assert(CalledFunc, "Special functions should all have a name");
  const auto &FuncName = CalledFunc->getName();

  StringToken Expression;

  if (FunctionTags::ModelGEP.isTagOf(CalledFunc)
      or FunctionTags::ModelGEPRef.isTagOf(CalledFunc)) {
    revng_assert(Call->getNumArgOperands() >= 2);

    bool IsRef = FunctionTags::ModelGEPRef.isTagOf(CalledFunc);

    // First argument is a string containing the base type
    auto *CurArg = Call->arg_begin();
    QualifiedType CurType = deserializeFromLLVMString(CurArg->get(), Model);

    // Second argument is the base llvm::Value
    ++CurArg;
    llvm::Value *BaseValue = CurArg->get();
    Expression = TokenMap.at(BaseValue);

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
        Expression = buildDerefExpr(Expression);
      } else {
        // Dereferencing a reference does not produce any code
      }
    } else {
      StringToken CurExpr = addParentheses(Expression);
      Tag DerefSymbol = IsRef ? operators::Dot : operators::Arrow;

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
            IndexExpr = constants::number(Const->getValue()).serialize();
          } else {
            IndexExpr = TokenMap.at(CurArg->get());
          }

          CurExpr += ("[" + IndexExpr + "]");
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
            CurExpr += ptml::getLocationReference(*Struct, Field);
            CurType = Struct->Fields().at(FieldIdx).Type();

          } else if (auto *Union = dyn_cast<model::UnionType>(UnqualType)) {
            const model::UnionField &Field = Union->Fields().at(FieldIdx);
            CurExpr += ptml::getLocationReference(*Union, Field);
            CurType = Union->Fields().at(FieldIdx).Type();

          } else {
            revng_abort("Unexpected ModelGEP type found: ");
            CurType.dump();
          }
        }

        // Regardless if the base type was a pointer or not, we are now
        // navigating only references
        DerefSymbol = operators::Dot;
      }

      Expression = CurExpr;
    }
  } else if (FunctionTags::ModelCast.isTagOf(CalledFunc)) {
    // First argument is a string containing the base type
    auto *CurArg = Call->arg_begin();
    QualifiedType CurType = deserializeFromLLVMString(CurArg->get(), Model);

    // Second argument is the base llvm::Value
    ++CurArg;
    llvm::Value *BaseValue = CurArg->get();

    // Emit the parenthesized cast expr, and we are done
    StringToken CastExpr = buildCastExpr(TokenMap.at(BaseValue),
                                         TypeMap.at(BaseValue),
                                         CurType);
    Expression = CastExpr;
  } else if (FunctionTags::AddressOf.isTagOf(CalledFunc)) {
    // First operand is the type of the value being addressed (should not
    // introduce casts)
    QualifiedType ArgType = deserializeFromLLVMString(Call->getArgOperand(0),
                                                      Model);

    // Second argument is the value being addressed
    llvm::Value *Arg = Call->getArgOperand(1);
    revng_assert(ArgType == TypeMap.at(Arg));
    revng_assert(TokenMap.contains(Arg));

    Expression = buildAddressExpr(TokenMap.at(Arg));

  } else if (FunctionTags::Parentheses.isTagOf(CalledFunc)) {
    Expression = addAlwaysParentheses(TokenMap.at(Call->getArgOperand(0)));

  } else if (FunctionTags::StructInitializer.isTagOf(CalledFunc)) {
    // Struct initializers should be used only to pack together return
    // values of RawFunctionTypes that return multiple values, therefore
    // they must have the same type as the function's return type
    auto *StructTy = cast<llvm::StructType>(Call->getType());
    revng_assert(Call->getFunction()->getReturnType() == StructTy);
    revng_assert(LLVMFunction.getReturnType() == StructTy);
    auto StrucTypeName = getNamedInstanceOfReturnType(ParentPrototype, "");
    StringToken StructInit = addAlwaysParentheses(StrucTypeName);

    // Emit RHS
    llvm::StringRef Separator = "{";
    for (const auto &Arg : Call->args()) {
      StructInit.append(Separator);
      StructInit.append(" ");
      StructInit.append(TokenMap.at(Arg));
      Separator = ",";
    }
    StructInit.append("}\n");

    Expression = StructInit;

  } else if (FunctionTags::LocalVariable.isTagOf(CalledFunc)
             or FuncName.startswith("revng_call_stack_arguments")) {
    const auto VarNames = createVarName(Call);

    // Declare a new local variable if it hasn't already been declared
    revng_assert(VarNames.hasDeclaration());
    Out << getNamedCInstance(TypeMap.at(Call), VarNames.Declaration) << ";\n";
    Expression = VarNames.Use;

  } else if (FuncName.startswith("revng_stack_frame")) {
    const auto VarNames = getOrCreateVarName(Call);

    // Declare a new local variable if it hasn't already been declared
    if (VarNames.hasDeclaration()) {
      Out << getNamedCInstance(TypeMap.at(Call), VarNames.Declaration) << ";\n";
    }

    Expression = VarNames.Use;
  } else if (FunctionTags::SegmentRef.isTagOf(CalledFunc)) {
    const auto &[StartAddress,
                 VirtualSize] = extractSegmentKeyFromMetadata(*CalledFunc);
    model::Segment Segment = Model.Segments().at({ StartAddress, VirtualSize });
    auto Name = Segment.name();

    Expression = ptml::getLocationReference(Segment);

  } else if (FunctionTags::Assign.isTagOf(CalledFunc)) {
    const llvm::Value *StoredVal = Call->getArgOperand(0);
    const llvm::Value *PointerVal = Call->getArgOperand(1);
    const QualifiedType PointedType = TypeMap.at(PointerVal);

    Expression = buildAssignmentExpr(PointedType,
                                     { TokenMap.at(PointerVal) },
                                     TokenMap.at(StoredVal));

  } else if (FunctionTags::Copy.isTagOf(CalledFunc)) {
    // Forward expression
    Expression = TokenMap.at(Call->getArgOperand(0));

  } else if (FunctionTags::OpaqueCSVValue.isTagOf(CalledFunc)) {
    std::string HelperRef = getHelperFunctionLocationReference(CalledFunc);
    Expression = buildFuncCallExpr(Call, HelperRef, /*prototype=*/nullptr);

  } else if (FunctionTags::QEMU.isTagOf(CalledFunc)
             or FunctionTags::Helper.isTagOf(CalledFunc)
             or FunctionTags::Exceptional.isTagOf(CalledFunc)
             or CalledFunc->isIntrinsic()) {

    revng_assert(not CalledFunc->isTargetIntrinsic());

    if (not Call->getType()->isVoidTy()) {
      const auto VarName = getOrCreateVarName(Call);
      if (VarName.hasDeclaration())
        Out << getReturnTypeLocationReference(CalledFunc) << " "
            << VarName.Declaration;
      else
        Out << VarName.Use;
      Out << " " << operators::Assign << " ";
      Expression = VarName.Use;
    }

    std::string HelperRef = getHelperFunctionLocationReference(CalledFunc);
    auto CallExpr = buildFuncCallExpr(Call, HelperRef, /*prototype=*/nullptr);
    Out << CallExpr << ";\n";

  } else if (FunctionTags::HexInteger.isTagOf(CalledFunc)) {
    const auto Operand = Call->getArgOperand(0);
    const auto *Value = cast<llvm::ConstantInt>(Operand);
    Expression = constants::constant(hexLiteral(Value)).serialize();
  } else if (FunctionTags::CharInteger.isTagOf(CalledFunc)) {
    const auto Operand = Call->getArgOperand(0);
    const auto *Value = cast<llvm::ConstantInt>(Operand);
    Expression = constants::constant(charLiteral(Value)).serialize();
  } else if (FunctionTags::BoolInteger.isTagOf(CalledFunc)) {
    const auto Operand = Call->getArgOperand(0);
    const auto *Value = cast<llvm::ConstantInt>(Operand);
    Expression = constants::constant(boolLiteral(Value)).serialize();
  } else if (FunctionTags::UnaryMinus.isTagOf(CalledFunc)) {
    auto Operand = Call->getArgOperand(0);
    Expression = operators::UnaryMinus
                 + constants::constant(TokenMap.at(Operand)).serialize();
  } else if (FunctionTags::BinaryNot.isTagOf(CalledFunc)) {
    auto Operand = Call->getArgOperand(0);
    Expression = (Operand->getType()->isIntegerTy(1) ? operators::BoolNot :
                                                       operators::BinaryNot)
                 + constants::constant(TokenMap.at(Operand)).serialize();
  } else {
    revng_abort("Unknown non-isolated function");
  }

  return Expression;
}

static bool shouldGenerateDebugInfoAsPTML(const llvm::Instruction &I) {
  if (!I.getDebugLoc() || !I.getDebugLoc()->getScope())
    return false;

  // If the next instruciton in the BB has different DebugLoc, generate the
  // PTML location now.
  auto NextInstr = std::next(I.getIterator());
  if (NextInstr == I.getParent()->end() || !NextInstr->getDebugLoc()
      || NextInstr->getDebugLoc() != I.getDebugLoc())
    return true;
  return false;
}

StringToken CCodeGenerator::buildExpression(const llvm::Instruction &I) {
  // Constants, ConstExprs and Globals might not have an associated token yet
  for (const llvm::Value *Operand : I.operand_values()) {
    if (not TokenMap.contains(Operand)) {
      bool Added = addOperandToken(Operand);
      if (Added) {
        decompilerLog(Out, "Added token: " + TokenMap.at(Operand));
      }
    }
  }

  StringToken FinalExpression;

  // Emit PTML debug locations, trying to avoid duplications
  ptml::Tag TheDebugInfoTag(ptml::tags::Span);
  bool ShouldGenerateDbgInfoPTML = shouldGenerateDebugInfoAsPTML(I);
  if (ShouldGenerateDbgInfoPTML) {
    TheDebugInfoTag.addAttribute(ptml::attributes::LocationReferences,
                                 I.getDebugLoc()->getScope()->getName());
    // Open the tag for debug location.
    FinalExpression += TheDebugInfoTag.open();
  }

  StringToken Expression;

  if (auto *Call = dyn_cast<llvm::CallInst>(&I)) {
    if (FunctionTags::CallToLifted.isTagOf(Call)) {
      revng_log(Log, "Emitting call to isolated function");

      // Retrieve the CallEdge
      const auto &[CallEdge, _] = Cache.getCallEdge(Model, Call);
      revng_assert(CallEdge);
      const auto &PrototypePath = Cache.getCallSitePrototype(Model, Call);

      // Construct the callee token (can be a function name or a function
      // pointer)
      StringToken CalleeToken;

      if (not isa<llvm::Function>(Call->getCalledOperand())) {
        // Indirect Call
        llvm::Value *CalledVal = Call->getCalledOperand();
        revng_assert(CalledVal);
        addOperandToken(CalledVal);
        auto &VarName = TokenMap.at(CalledVal);

        CalleeToken = addParentheses(VarName);

      } else {
        if (not CallEdge->DynamicFunction().empty()) {
          // Dynamic Function
          auto &DynFuncID = CallEdge->DynamicFunction();
          auto &DynamicFunc = Model.ImportedDynamicFunctions().at(DynFuncID);
          std::string Location = serializedLocation(ranks::DynamicFunction,
                                                    DynamicFunc.key());
          CalleeToken = Tag(tags::Span, DynamicFunc.name().str())
                          .addAttribute(attributes::Token, tokens::Function)
                          .addAttribute(attributes::ModelEditPath,
                                        getCustomNamePath(DynamicFunc))
                          .addAttribute(attributes::LocationReferences,
                                        Location)
                          .serialize();
        } else {
          // Isolated function
          llvm::Function *CalledFunc = Call->getCalledFunction();
          revng_assert(CalledFunc);
          const model::Function *ModelFunc = llvmToModelFunction(Model,
                                                                 *CalledFunc);
          revng_assert(ModelFunc);
          CalleeToken = ModelFunc->name();
          CalleeToken = Tag(tags::Span, ModelFunc->name().str())
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
      Expression = buildFuncCallExpr(Call, CalleeToken, Prototype);

      const auto Layout = abi::FunctionType::Layout::make(*Prototype);
      if (Layout.ReturnValues.size() > 1) {
        revng_assert(Call->getType()->isAggregateType());
        // When functions return multiple values, we need to insert a struct
        // to represent those.
        //
        // The name of such a struct should be derived immediately as postponing
        // the emission to the next `AssignmentMarker` would result in the loss
        // of the name of this struct.
        const auto VarNames = createVarName(Call);

        // Declare a new local variable if it hasn't already been declared
        if (VarNames.hasDeclaration())
          Out << getNamedInstanceOfReturnType(*Prototype, VarNames.Declaration);
        else
          Out << VarNames.Use;

        Out << " " << operators::Assign << " " << Expression << ";\n";
        Expression = VarNames.Use;
      } else if (Layout.ReturnValues.size() == 1) {
        // Similarly to multiple return values, when returning an array, it
        // should be enclosed into a struct.
        if (Layout.ReturnValues.front().Type.isArray()) {
          revng_assert(llvm::isa<CABIFunctionType>(Prototype),
                       "Array return values are only supported when "
                       "the function has `CABIFunctionType`.");

          const auto VarNames = createVarName(Call);

          // Declare a new local variable if it hasn't already been declared
          if (VarNames.hasDeclaration())
            Out << getNamedInstanceOfReturnType(*Prototype,
                                                VarNames.Declaration);
          else
            Out << VarNames.Use;

          Out << " " << operators::Assign << " " << Expression << ";\n";
          Expression = VarNames.Use;
        }
      }
    } else {
      // Non-lifted function calls have to be dealt with separately
      Expression = handleSpecialFunction(Call);
    }

  } else if (auto *Load = dyn_cast<llvm::LoadInst>(&I)) {
    const llvm::Value *Pointer = Load->getPointerOperand();
    // The pointer operand's type and the actual loaded value's type might have
    // a mismatch. In this case, we want to cast the pointer operand to correct
    // type pointer before dereferencing it.
    const model::Architecture::Values &Architecture = Model.Architecture();
    QualifiedType ResultPtrType = TypeMap.at(Load).getPointerTo(Architecture);
    Expression = (buildDerefExpr(buildCastExpr(TokenMap.at(Pointer),
                                               TypeMap.at(Pointer),
                                               ResultPtrType)))
                   .str();

  } else if (auto *Store = dyn_cast<llvm::StoreInst>(&I)) {

    const llvm::Value *PointerOp = Store->getPointerOperand();
    const llvm::Value *ValueOp = Store->getValueOperand();
    const QualifiedType &StoredType = TypeMap.at(ValueOp);

    const model::Architecture::Values &Architecture = Model.Architecture();
    const auto PointerToStoredType = StoredType.getPointerTo(Architecture);
    StringToken PointerCast = buildCastExpr(TokenMap.at(PointerOp),
                                            TypeMap.at(PointerOp),
                                            PointerToStoredType);

    Expression = buildAssignmentExpr(StoredType,
                                     { buildDerefExpr(PointerCast) },
                                     TokenMap.at(ValueOp));

  } else if (auto *Select = dyn_cast<llvm::SelectInst>(&I)) {

    StringToken Condition = StringToken(TokenMap.at(Select->getCondition()));
    const llvm::Value *Op1 = Select->getOperand(1);
    const llvm::Value *Op2 = Select->getOperand(2);

    StringToken Op1Token = buildCastExpr(TokenMap.at(Op1),
                                         TypeMap.at(Op1),
                                         TypeMap.at(Select));
    StringToken Op2Token = buildCastExpr(TokenMap.at(Op2),
                                         TypeMap.at(Op2),
                                         TypeMap.at(Select));

    Expression = (addParentheses(Condition) + " ? " + addParentheses(Op1Token)
                  + " : " + addParentheses(Op2Token))
                   .str();

  } else if (auto *Alloca = dyn_cast<llvm::AllocaInst>(&I)) {
    auto [VarName, VarDeclaration] = createVarName(Alloca);
    if (!VarDeclaration.empty()) {
      // Declare a local variable
      auto AllocaDeclaration = declareAllocaVariable(Alloca,
                                                     { VarName,
                                                       VarDeclaration });
      Expression = AllocaDeclaration.Use;
    } else {
      // If it's a top-scope variable it has already been declared, so we have
      // nothing left to do
      Expression = VarName;
    }

  } else if (auto *Ret = dyn_cast<llvm::ReturnInst>(&I)) {
    Expression = keywords::Return.serialize();

    if (llvm::Value *ReturnedVal = Ret->getReturnValue())
      Expression += (" " + TokenMap.at(ReturnedVal));

  } else if (auto *Branch = dyn_cast<llvm::BranchInst>(&I)) {
    // This is never emitted directly in the BB: it is used when
    // emitting control-flow statements during the GHAST visit.
  } else if (auto *Switch = dyn_cast<llvm::SwitchInst>(&I)) {
    // This is never emitted directly in the BB: it is used when emitting
    // control-flow statements during the GHAST visit
  } else if (isa<llvm::IntToPtrInst>(&I) or isa<llvm::PtrToIntInst>(&I)) {
    // Pointer <-> Integer casts are transparent, since the distinction
    // between integers and pointers is left to the model to decide
    const llvm::Value *Operand = I.getOperand(0);
    Expression = TokenMap.at(Operand);

  } else if (auto *Bin = dyn_cast<llvm::BinaryOperator>(&I)) {
    const llvm::Value *Op1 = Bin->getOperand(0);
    const llvm::Value *Op2 = Bin->getOperand(1);
    const QualifiedType &ResultType = TypeMap.at(Bin);

    const auto &Op1Token = buildCastExpr(TokenMap.at(Op1),
                                         TypeMap.at(Op1),
                                         ResultType);
    const auto &Op2Token = buildCastExpr(TokenMap.at(Op2),
                                         TypeMap.at(Op2),
                                         ResultType);

    // TODO: Integer promotion
    Expression = (addParentheses(Op1Token) + getBinOpString(Bin)
                  + addParentheses(Op2Token))
                   .str();

  } else if (auto *Cmp = dyn_cast<llvm::CmpInst>(&I)) {
    const llvm::Value *Op1 = Cmp->getOperand(0);
    const llvm::Value *Op2 = Cmp->getOperand(1);
    const QualifiedType &ResultType = llvmIntToModelType(Op1->getType(), Model);

    const auto &Op1Token = buildCastExpr(TokenMap.at(Op1),
                                         TypeMap.at(Op1),
                                         ResultType);
    const auto &Op2Token = buildCastExpr(TokenMap.at(Op2),
                                         TypeMap.at(Op2),
                                         ResultType);

    // TODO: Integer promotion
    Expression = (addParentheses(Op1Token) + getCmpOpString(Cmp->getPredicate())
                  + addParentheses(Op2Token))
                   .str();

  } else if (auto *Cast = dyn_cast<llvm::CastInst>(&I)) {

    const llvm::Value *Op = Cast->getOperand(0);
    Expression = buildCastExpr(TokenMap.at(Op),
                               TypeMap.at(Op),
                               TypeMap.at(Cast));

  } else if (auto *ExtractVal = dyn_cast<llvm::ExtractValueInst>(&I)) {

    // Note: ExtractValues at this point should have been already
    // handled when visiting the instruction that generated their
    // struct operand
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
      StructFieldRef = getReturnStructFieldLocationReference(Callee, Idx);
    } else {
      const model::Type *CalleeType = CalleePrototype.getConst();
      StructFieldRef = getReturnField(*CalleeType, Idx, Model).str().str();
    }

    Expression.assign(TokenMap.at(AggregateOp) + "." + StructFieldRef);

  } else if (auto *Unreach = dyn_cast<llvm::UnreachableInst>(&I)) {
    Expression = "__builtin_unreachable()";

  } else {
    revng_abort("Unexpected instruction found when decompiling");
  }

  if (Expression.empty())
    return Expression;

  // Clear the TokenMap for operands that only have one use in the same
  // BasicBlock, and such that I is the last user in the block.
  // They will be never used after that, and we don't want those strings to hang
  // around, since we don't want the container to grow too big.
  for (const llvm::Value *Operand : I.operand_values()) {
    if (auto *InstructionOp = dyn_cast<llvm::Instruction>(Operand)) {
      bool UserInDifferentBlock = false;
      for (auto *User : InstructionOp->users())
        if (auto *UserInst = cast<llvm::Instruction>(User))
          if (UserInst->getParent() != InstructionOp->getParent())
            UserInDifferentBlock = true;

      if (not UserInDifferentBlock) {
        llvm::SmallPtrSet<const llvm::Instruction *, 8> UserInstructions;
        for (const auto *User : Operand->users())
          UserInstructions.insert(cast<llvm::Instruction>(User));

        bool FoundOtherUser = false;
        for (const auto &I :
             llvm::make_range(std::next(I.getIterator()), I.getParent()->end()))
          if (UserInstructions.contains(&I))
            FoundOtherUser = true;

        if (not FoundOtherUser)
          TokenMap.erase(Operand);
      }
    }
  }

  FinalExpression += Expression;

  // Close the tag for debug location.
  if (ShouldGenerateDbgInfoPTML)
    FinalExpression += TheDebugInfoTag.close();

  return FinalExpression;
}

void CCodeGenerator::emitBasicBlock(const llvm::BasicBlock *BB) {
  LoggerIndent Indent{ VisitLog };
  revng_log(VisitLog, "|__ Visiting BB " << BB->getName());
  LoggerIndent MoreIndent{ VisitLog };
  revng_log(Log, "--------- BB " << BB->getName());

  for (const Instruction &I : *BB) {
    // Guard this checking logger to prevent computing dumpToString if loggers
    // are not enabled.
    if (Log.isEnabled() or InlineLog.isEnabled()) {
      revng_log(Log, "+++++++++");
      decompilerLog(Out, "Analyzing: " + dumpToString(I));
    }

    StringToken Expression = buildExpression(I);

    if (not Expression.empty()) {
      if (I.getType()->isVoidTy()) {
        decompilerLog(Out, "Void instruction found: serializing expression");
        Out << Expression << ";\n";
      } else {
        decompilerLog(Out,
                      "Adding expression to the TokenMap " + Expression.str());
        TokenMap[&I] = Expression.str().str();
      }
    } else {
      decompilerLog(Out, "Nothing to serialize");
    }
  }
}

RecursiveCoroutine<StringToken>
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
    rc_return StringToken(TokenMap.at(Br->getCondition()));
  } break;

  case NodeKind::NK_Not: {
    revng_log(VisitLog, "(not)");

    const NotNode *N = cast<NotNode>(E);
    ExprNode *Negated = N->getNegatedNode();

    StringToken Expression;
    Expression = operators::BoolNot
                 + addAlwaysParentheses(rc_recur buildGHASTCondition(Negated));
    rc_return Expression;
  } break;

  case NodeKind::NK_And:
  case NodeKind::NK_Or: {
    revng_log(VisitLog, "(and/or)");

    const BinaryNode *Binary = cast<BinaryNode>(E);

    const auto &[Child1, Child2] = Binary->getInternalNodes();
    const auto Child1Token = rc_recur buildGHASTCondition(Child1);
    const auto Child2Token = rc_recur buildGHASTCondition(Child2);
    const Tag &OpToken = E->getKind() == NodeKind::NK_And ? operators::BoolAnd :
                                                            operators::BoolOr;
    StringToken Expression = addAlwaysParentheses(Child1Token);
    Expression += " " + OpToken + " ";
    Expression += addAlwaysParentheses(Child2Token);
    rc_return Expression;
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
    if (Break->breaksFromWithinSwitch()) {
      revng_assert(not SwitchStateVars.empty()
                   and not SwitchStateVars.back().empty());
      Out << SwitchStateVars.back()
          << " " + operators::Assign + " " + constants::True + ";\n";
    }
  };
    [[fallthrough]];

  case ASTNode::NodeKind::NK_SwitchBreak: {
    revng_log(VisitLog, "(NK_SwitchBreak)");

    Out << keywords::Break << ";\n";
  } break;

  case ASTNode::NodeKind::NK_Continue: {
    revng_log(VisitLog, "(NK_Continue)");

    const ContinueNode *Continue = cast<ContinueNode>(N);

    // Print the condition computation code of the if statement.
    if (Continue->hasComputation()) {
      IfNode *ComputationIfNode = Continue->getComputationIfNode();
      buildGHASTCondition(ComputationIfNode->getCondExpr());
    }

    // Actually print the continue statement only if the continue is not
    // implicit (i.e. it is not the last statement of the loop).
    if (not Continue->isImplicit())
      Out << keywords::Continue << ";\n";
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
    const StringToken CondExpr = buildGHASTCondition(If->getCondExpr());
    // "If" expression
    // TODO: possibly cast the CondExpr if it's not convertible to boolean?
    revng_assert(not CondExpr.empty());
    Out << keywords::If << " (" + CondExpr + ") ";
    {
      Scope TheScope(Out);
      // "Then" expression (always emitted)
      if (nullptr == If->getThen())
        Out << helpers::lineComment("Empty");
      else
        rc_recur emitGHASTNode(If->getThen());
    }

    // "Else" expression (optional)
    if (If->hasElse()) {
      Out << " " + keywords::Else + " ";
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
    StringToken CondExpr(constants::True.serialize());
    if (LoopBody->isDoWhile() or LoopBody->isWhile()) {
      const IfNode *LoopCondition = LoopBody->getRelatedCondition();
      revng_assert(LoopCondition);

      // Retrieve the expression of the condition as well as emitting its
      // associated basic block
      CondExpr = buildGHASTCondition(LoopCondition->getCondExpr());
      revng_assert(not CondExpr.empty());
    }

    if (LoopBody->isDoWhile())
      Out << keywords::Do << " ";
    else
      Out << keywords::While + " (" + CondExpr + ") ";

    revng_assert(LoopBody->hasBody());
    {
      Scope TheScope(Out);
      rc_recur emitGHASTNode(LoopBody->getBody());
    }

    if (LoopBody->isDoWhile())
      Out << " " + keywords::While + " (" + CondExpr + ");";
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
      std::string SwitchStateVar = getVariableLocationReference(NewVarName,
                                                                ModelFunction);
      SwitchStateVars.push_back(std::move(SwitchStateVar));
      Out << ptml::tokenTag("bool", tokens::Type) << " "
          << getVariableLocationDefinition(NewVarName, ModelFunction)
          << " " + operators::Assign + " " + constants::False + ";\n";
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
      SwitchVarToken = TokenMap.at(SwitchVar);
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
    Out << keywords::Switch + " (" << SwitchVarToken << ") ";
    {
      Scope TheScope(Out);

      // Generate the body of the switch (except for the default)
      for (const auto &[Labels, CaseNode] : Switch->cases_const_range()) {
        revng_assert(not Labels.empty());
        // Generate the case label(s) (multiple case labels might share the
        // same body)
        for (uint64_t CaseVal : Labels) {
          Out << keywords::Case + " ";
          if (SwitchVar) {
            llvm::Type *SwitchVarT = SwitchVar->getType();
            auto *IntType = cast<llvm::IntegerType>(SwitchVarT);
            auto *CaseConst = llvm::ConstantInt::get(IntType, CaseVal);
            // TODO: assigned the signedness based on the signedness of the
            // condition
            Out << constants::number(CaseConst->getValue());
          } else {
            Out << constants::number(CaseVal);
          }
          Out << ":\n";
        }

        {
          Scope InnerScope(Out);
          // Generate the case body
          rc_recur emitGHASTNode(CaseNode);
        }
        Out << " " + keywords::Break + ";\n";
      }

      // Generate the default case if it exists
      if (auto *Default = Switch->getDefault()) {
        Out << keywords::Default << ":\n";
        {
          Scope TheScope(Out);
          rc_recur emitGHASTNode(Default);
        }
        Out << " " + keywords::Break + ";\n";
      }
    }
    Out << "\n";

    // If the switch needs a loop break dispatcher, reset the associated
    // state variable before emitting the switch statement.
    if (Switch->needsLoopBreakDispatcher()) {
      revng_assert(not SwitchStateVars.empty()
                   and not SwitchStateVars.back().empty());
      Out << keywords::If + " (" + SwitchStateVars.back() + ")";
      {
        auto Scope = scopeTags::Scope.scope(Out, true);
        auto IndentScope = Out.scope();
        Out << keywords::Break + ";";
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
    Out << LoopStateVar << " " << operators::Assign << " " << StateValue
        << ";\n";
  } break;
  }

  rc_return;
}

void CCodeGenerator::emitFunction(bool NeedsLocalStateVar) {
  revng_log(Log, "========= Emitting Function " << LLVMFunction.getName());
  revng_log(VisitLog, "========= Function " << LLVMFunction.getName());
  LoggerIndent Indent{ VisitLog };

  // Create a token for each of the function's arguments
  if (auto *RawPrototype = dyn_cast<model::RawFunctionType>(&ParentPrototype)) {

    const auto &ModelArgs = RawPrototype->Arguments();
    const auto &LLVMArgs = LLVMFunction.args();
    const auto LLVMArgsNum = LLVMFunction.arg_size();

    const auto &StackArgType = RawPrototype->StackArgumentsType()
                                 .UnqualifiedType();
    const auto ArgSize = ModelArgs.size();
    revng_assert(LLVMArgsNum == ArgSize
                 or (LLVMArgsNum == ArgSize + 1 and StackArgType.isValid()));

    // Associate each LLVM argument with its name
    for (const auto &[ModelArg, LLVMArg] :
         llvm::zip_first(ModelArgs, LLVMArgs)) {
      revng_log(Log, "Adding token for: " << dumpToString(LLVMArg));
      std::string ArgIdentifier = model::Identifier::fromString(ModelArg.name())
                                    .str()
                                    .str();
      TokenMap[&LLVMArg] = getArgumentLocationReference(ArgIdentifier,
                                                        ModelFunction);
    }

    // Add a token for the stack arguments
    if (StackArgType.isValid()) {
      const auto *LLVMArg = LLVMFunction.getArg(LLVMArgsNum - 1);
      revng_log(Log, "Adding token for: " << dumpToString(LLVMArg));
      TokenMap[LLVMArg] = getArgumentLocationReference("stack_args",
                                                       ModelFunction);
    }
  } else if (auto *CPrototype = dyn_cast<CABIFunctionType>(&ParentPrototype)) {

    const auto &ModelArgs = CPrototype->Arguments();
    const auto &LLVMArgs = LLVMFunction.args();

    revng_assert(LLVMFunction.arg_size() == ModelArgs.size());

    // Associate each LLVM argument with its name
    for (const auto &[ModelArg, LLVMArg] : llvm::zip(ModelArgs, LLVMArgs)) {
      revng_log(Log, "Adding token for: " << dumpToString(LLVMArg));
      std::string ArgIdentifier = model::Identifier::fromString(ModelArg.name())
                                    .str()
                                    .str();
      TokenMap[&LLVMArg] = getArgumentLocationReference(ArgIdentifier,
                                                        ModelFunction);
    }
  } else {
    revng_abort("Functions can only have RawFunctionType or "
                "CABIFunctionType.");
  }

  {
    auto FunctionTagScope = scopeTags::Function.scope(Out);
    // Print function's prototype
    printFunctionPrototype(ParentPrototype, ModelFunction, Out, Model, true);
    Out << " ";
    {
      Scope BraceScope(Out, scopeTags::FunctionBody);

      if (not TopScopeVariables.empty()) {
        // Declare all variables that have the entire function as a scope
        decompilerLog(Out, "Top-Scope Declarations");
        for (const llvm::Instruction *VarToDeclare : TopScopeVariables) {
          if (Log.isEnabled() or InlineLog.isEnabled()) {
            decompilerLog(Out, "VarToDeclare: " + dumpToString(VarToDeclare));
          }

          VariableTokens VarName = createVarName(VarToDeclare);

          auto VarTypeIt = TypeMap.find(VarToDeclare);
          if (VarTypeIt != TypeMap.end()) {
            if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(VarToDeclare)) {
              // Allocas are special, since the expression associated to them is
              // `&var` while the variable allocated is `var`
              VarName = declareAllocaVariable(Alloca, VarName);
            } else {
              Out << getNamedCInstance(TypeMap.at(VarToDeclare),
                                       VarName.Declaration)
                  << ";\n";
            }

          } else {
            // The only types that are allowed to be missing from the TypeMap
            // are LLVM aggregates returned by RawFunctionTypes or by helpers
            auto *Call = llvm::cast<CallInst>(VarToDeclare);
            if (const auto &Prototype = Cache.getCallSitePrototype(Model, Call);
                Prototype.isValid() and not Prototype.empty()) {
              const auto *FunctionType = Prototype.getConst();
              Out << getNamedInstanceOfReturnType(*FunctionType,
                                                  VarName.Declaration)
                  << ";\n";
            } else {
              auto *CalledFunction = Call->getCalledFunction();
              revng_assert(CalledFunction);
              Out << getReturnTypeLocationReference(CalledFunction) << " "
                  << VarName.Declaration << ";\n";
            }
          }

          TokenMap[VarToDeclare] = VarName.Use.str().str();
        }
        decompilerLog(Out, "End of Top-Scope Declarations");
      }

      // Emit a declaration for the loop state variable, which is used to
      // redirect control flow inside loops (e.g. if we want to jump in the
      // middle of a loop during a certain iteration)
      if (NeedsLocalStateVar)
        Out << ptml::tokenTag("uint64_t", tokens::Type) << " "
            << LoopStateVarDeclaration << ";\n";

      // Recursively print the body of this function
      emitGHASTNode(GHAST.getRoot());
    }
  }
  Out << "\n";
}

static std::string decompileFunction(FunctionMetadataCache &Cache,
                                     const llvm::Function &LLVMFunc,
                                     const ASTTree &CombedAST,
                                     const Binary &Model,
                                     const InstrSetVec &TopScopeVariables,
                                     bool NeedsLocalStateVar) {
  std::string Result;

  llvm::raw_string_ostream Out(Result);
  CCodeGenerator Backend(Cache,
                         Model,
                         LLVMFunc,
                         CombedAST,
                         TopScopeVariables,
                         Out);
  Backend.emitFunction(NeedsLocalStateVar);
  Out.flush();

  return Result;
}

using Container = revng::pipes::DecompiledCCodeInYAMLStringMap;
void decompile(FunctionMetadataCache &Cache,
               llvm::Module &Module,
               const model::Binary &Model,
               Container &DecompiledFunctions) {
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
      const llvm::Twine &ASTFileName = F.getName()
                                       + "GHAST-during-c-codegen.dot";
      GHAST.dumpASTOnFile(ASTFileName.str());
    }

    // Generated C code for F
    auto TopScopeVariables = collectTopScopeVariables(F);
    auto NeedsLoopStateVar = hasLoopDispatchers(GHAST);
    std::string CCode = decompileFunction(Cache,
                                          F,
                                          GHAST,
                                          Model,
                                          TopScopeVariables,
                                          NeedsLoopStateVar);

    // Push the C code into
    MetaAddress Key = getMetaAddressMetadata(&F, "revng.function.entry");
    DecompiledFunctions.insert_or_assign(Key, std::move(CCode));
  }
}
