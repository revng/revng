//
// Copyright rev.ng Labs Srl. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
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
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/EarlyFunctionAnalysis/IRHelpers.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Generated/Early/PrimitiveTypeKind.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Qualifier.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Model/StructType.h"
#include "revng/Model/Type.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/BeautifyGHAST/BeautifyGHAST.h"
#include "revng-c/InitModelTypes/InitModelTypes.h"
#include "revng-c/RestructureCFG/ASTNode.h"
#include "revng-c/RestructureCFG/ASTTree.h"
#include "revng-c/RestructureCFG/RestructureCFG.h"
#include "revng-c/Support/FunctionTags.h"
#include "revng-c/Support/IRHelpers.h"
#include "revng-c/Support/ModelHelpers.h"
#include "revng-c/TypeNames/LLVMTypeNames.h"
#include "revng-c/TypeNames/ModelTypeNames.h"

#include "VariableScopeAnalysis.h"

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

using StringToken = llvm::SmallString<32>;
using TokenMapT = std::map<const llvm::Value *, StringToken>;
using ModelTypesMap = std::map<const llvm::Value *, const model::QualifiedType>;
using ValueSet = llvm::SmallPtrSet<const llvm::Value *, 32>;

static constexpr const char *StackFrameVarName = "stack";
static constexpr const char *StackPtrVarName = "stack_ptr";

static Logger<> Log{ "c-backend" };
static Logger<> InlineLog{ "c-backend-inline" };

/// Helper function that also writes the logged string as a comment in the C
/// file if the corresponding logger is enabled
static void decompilerLog(llvm::raw_ostream &Out, const llvm::Twine &Expr) {
  revng_log(Log, Expr.str());
  if (InlineLog.isEnabled())
    Out << "/* " << Expr << " */\n";
}

/// Traverse all nested typedefs inside \a QT, if any, and merge them into the
/// top type.
static void flattenTypedefs(QualifiedType &QT) {
  while (auto *TD = dyn_cast<TypedefType>(QT.UnqualifiedType.getConst())) {
    llvm::copy(TD->UnderlyingType.Qualifiers, QT.Qualifiers.begin());
    QT.UnqualifiedType = TD->UnderlyingType.UnqualifiedType;
  }
}

static void keepOnlyPtrAndArrayQualifiers(QualifiedType &QT) {
  llvm::erase_if(QT.Qualifiers, [](model::Qualifier &Q) {
    return not(model::Qualifier::isPointer(Q) or model::Qualifier::isArray(Q));
  });
}

/// Return the string that represents the given binary operator in C
static const char *getBinOpString(const llvm::BinaryOperator *BinOp) {
  switch (BinOp->getOpcode()) {
  case Instruction::Add:
    return " + ";
    break;
  case Instruction::Sub:
    return " - ";
    break;
  case Instruction::Mul:
    return " * ";
    break;
  case Instruction::SDiv:
  case Instruction::UDiv:
    return " / ";
    break;
  case Instruction::SRem:
  case Instruction::URem:
    return " % ";
    break;
  case Instruction::LShr:
  case Instruction::AShr:
    return " >> ";
    break;
  case Instruction::Shl:
    return " << ";
    break;
  case Instruction::And:
    return " & ";
    break;
  case Instruction::Or:
    return " | ";
    break;
  case Instruction::Xor:
    return " ^ ";
    break;
  default:
    revng_abort("Unknown const Binary operation");
  }
}

/// Return the string that represents the given comparison operator in C
static const char *getCmpOpString(const llvm::CmpInst::Predicate &Pred) {
  using llvm::CmpInst;
  switch (Pred) {
  case CmpInst::ICMP_EQ: ///< equal
    return " == ";
    break;
  case CmpInst::ICMP_NE: ///< not equal
    return " != ";
    break;
  case CmpInst::ICMP_UGT: ///< unsigned greater than
  case CmpInst::ICMP_SGT: ///< signed greater than
    return " > ";
    break;

  case CmpInst::ICMP_UGE: ///< unsigned greater or equal
  case CmpInst::ICMP_SGE: ///< signed greater or equal
    return " >= ";
    break;

  case CmpInst::ICMP_ULT: ///< unsigned less than
  case CmpInst::ICMP_SLT: ///< signed less than
    return " < ";
    break;

  case CmpInst::ICMP_ULE: ///< unsigned less or equal
  case CmpInst::ICMP_SLE: ///< signed less or equal
    return " <= ";
    break;

  default:
    revng_abort("Unknown comparison operator");
  }
}

static StringToken addParentheses(const llvm::Twine &Expr) {
  return StringToken(("(" + Expr + ")").str());
}

static StringToken buildDerefExpr(const llvm::Twine &Expr) {
  return StringToken(("*" + addParentheses(Expr)).str());
}

static StringToken buildAddressExpr(const llvm::Twine &Expr) {
  return StringToken(("&" + addParentheses(Expr)).str());
}

/// Return a C string that represents a cast of \a ExprToCast to a given
/// \a DestType. If no casting is needed between the two expression, the
/// original expression is returned.
static StringToken buildCastExpr(StringRef ExprToCast,
                                 const model::QualifiedType &SrcType,
                                 const model::QualifiedType &DestType) {
  if (SrcType == DestType or not SrcType.UnqualifiedType.isValid()
      or not SrcType.UnqualifiedType.isValid())
    return ExprToCast;

  StringToken CastString;

  if ((SrcType.isScalar() or SrcType.isPointer())
      and (DestType.isScalar() or DestType.isPointer()))
    CastString = (addParentheses(getTypeName(DestType))
                  + addParentheses(ExprToCast))
                   .str();
  else
    CastString = buildDerefExpr(addParentheses(getTypeName(DestType) + " *")
                                + buildAddressExpr(ExprToCast));

  return CastString;
}

/// Return a string that represents a C assignment:
///
///         [<TYPE>] LHSToken = [<CAST>] RHSToken
///
/// \note <TYPE> is added if \a WithDeclaration is true (transforms the
/// assignment into a declaration + assignment string)
/// \note <CAST> is automatically added if LHSType and RHSType are different
static StringToken buildAssignmentExpr(const model::QualifiedType &LHSType,
                                       const llvm::StringRef &LHSToken,
                                       const model::QualifiedType &RHSType,
                                       const llvm::StringRef &RHSToken,
                                       bool WithDeclaration) {

  StringToken AssignmentStr;

  if (WithDeclaration)
    AssignmentStr += getNamedCInstance(LHSType, LHSToken);
  else
    AssignmentStr += LHSToken;

  AssignmentStr += " = ";
  AssignmentStr += buildCastExpr(RHSToken, LHSType, RHSType);

  return AssignmentStr;
}

/// Stateful name assignment for local variables.
class VarNameGenerator {
private:
  uint64_t CurVarID = 0;

  StringToken nextFuncPtrName() {
    StringToken FuncName("func_ptr_");
    FuncName += to_string(CurVarID++);
    return FuncName;
  }

public:
  StringToken nextVarName() {
    StringToken VarName("var_");
    VarName += to_string(CurVarID++);
    return VarName;
  }

  StringToken nextVarName(const llvm::Value *V) {
    if (V->getType()->isPointerTy()) {
      const auto *PtrType = llvm::cast<llvm::PointerType>(V->getType());
      const auto *PointedType = PtrType->getElementType();
      if (PointedType->isFunctionTy())
        return nextFuncPtrName();
    }

    return nextVarName();
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
  const ValueSet &TopScopeVariables;
  /// A map containing a model type for each LLVM value in the function
  const ModelTypesMap TypeMap;

  /// Where to output the decompiled C code
  raw_ostream &Out;

  /// Name of the local variable used to break out of loops from within nested
  /// switches
  llvm::SmallVector<StringToken> SwitchStateVars;

private:
  /// Stateful generator for variable names
  VarNameGenerator NameGenerator;
  /// Keep track of which expression is associated to each LLVM value during the
  /// emission of C code
  TokenMapT TokenMap;

private:
  /// Name of the local variable used to break out from loops
  StringToken LoopStateVar;

private:
  /// During emission, keep track of the values that already have a declaration
  llvm::SmallPtrSet<const llvm::Value *, 8> AlreadyDeclared;

public:
  CCodeGenerator(const Binary &Model,
                 const llvm::Function &LLVMFunction,
                 const ASTTree &GHAST,
                 const ValueSet &TopScopeVariables,
                 raw_ostream &Out) :
    Model(Model),
    LLVMFunction(LLVMFunction),
    ModelFunction(*llvmToModelFunction(Model, LLVMFunction)),
    ParentPrototype(*ModelFunction.Prototype.getConst()),
    GHAST(GHAST),
    TopScopeVariables(TopScopeVariables),
    TypeMap(initModelTypes(LLVMFunction,
                           &ModelFunction,
                           Model,
                           /*PointersOnly=*/false)),
    Out(Out),
    SwitchStateVars() {

    // TODO: don't use a global loop state variable
    LoopStateVar = "loop_state_var";
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
  /// Emit an assignment for an instruction, if it is marked for assignemnt,
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
};

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

    for (auto &ArgEnum : llvm::enumerate(Call->arg_operands())) {
      Expression += Separator;

      model::QualifiedType FormalArgType;
      if (Prototype) {
        // If we have the function's model prototype, we can infer the formal
        // argument type from it
        if (auto *RawPrototype = dyn_cast<RawFunctionType>(Prototype)) {
          const auto ModelArgSize = RawPrototype->Arguments.size();
          const auto ArgIdx = ArgEnum.index();

          if (ArgIdx < ModelArgSize) {
            auto ModelArg = RawPrototype->Arguments.begin() + ArgEnum.index();
            FormalArgType = ModelArg->Type;

          } else if (ArgIdx == ModelArgSize) {
            // If the LLVM argument is past the end of the model arguments list,
            // it's a stack argument: create a pointer to the
            // stackArgumentsType.
            QualifiedType StackArgsType = RawPrototype->StackArgumentsType;
            addPointerQualifier(StackArgsType, Model);
            revng_assert(StackArgsType.UnqualifiedType.isValid());
            FormalArgType = StackArgsType;

          } else {
            revng_abort("Out-of-bounds access to model arguments");
          }

        } else if (auto *CPrototype = dyn_cast<CABIFunctionType>(Prototype)) {
          auto ModelArg = CPrototype->Arguments.begin() + ArgEnum.index();
          FormalArgType = ModelArg->Type;
        }
      } else {
        // If we don't have the model prototype, we inspect the LLVM prototype
        // and convert the argument's LLVM type to a QUalifiedType
        const llvm::Function *CalledF = Call->getCalledFunction();
        revng_assert(CalledF);
        const llvm::Argument *LLVMArg = CalledF->getArg(ArgEnum.index());

        FormalArgType = llvmIntToModelType(LLVMArg->getType(), Model);
      }

      const llvm::Value *Arg = ArgEnum.value();
      Expression += buildCastExpr(TokenMap.at(Arg),
                                  TypeMap.at(Arg),
                                  FormalArgType);
      Separator = ", ";
    }
    Expression += ")";
  }

  return Expression;
}

RecursiveCoroutine<bool>
CCodeGenerator::addOperandToken(const llvm::Value *Operand) {
  revng_log(Log, "\tOperand: " << dumpToString(*Operand));

  // Instructions must be visited in reverse-postorder when filling the
  // TokenMap
  if (isa<llvm::Instruction>(Operand) or isa<llvm::Argument>(Operand)) {
    revng_assert(TokenMap.contains(Operand));
    rc_return false;
  }

  revng_assert(not Operand->getType()->isVoidTy());

  if (auto *Const = dyn_cast<llvm::ConstantInt>(Operand)) {
    llvm::APInt Value = Const->getValue();
    if (Value.isIntN(64)) {
      // TODO: Decide how to print constants
      Value.toString(TokenMap[Operand],
                     /*radix=*/10,
                     /*signed=*/false,
                     /*formatAsCLiteral=*/true);
    } else {
      // In C, even if you can have 128-bit variables, you cannot have 128-bit
      // literals, so we need this hack to assign a big constant value to a
      // 128-bit variable.
      llvm::APInt LowBits = Value.getLoBits(64);
      llvm::APInt HighBits = Value.getHiBits(Value.getBitWidth() - 64);

      // TODO: Decide how to print constants
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

      auto CompositeConstant = addParentheses(HighBitsString + " << 64") + " | "
                               + LowBitsString;
      TokenMap[Operand] = addParentheses(CompositeConstant).str();
    }

  } else if (auto *Null = dyn_cast<llvm::ConstantPointerNull>(Operand)) {
    TokenMap[Operand] = "NULL";

  } else if (auto *Glob = dyn_cast<llvm::GlobalVariable>(Operand)) {
    TokenMap[Operand] = Glob->getNameOrAsOperand();

  } else if (auto *ConstExpr = dyn_cast<llvm::ConstantExpr>(Operand)) {
    // A constant expression might have its own uninitialized constant operands
    for (const llvm::Value *Op : ConstExpr->operand_values())
      rc_recur addOperandToken(Op);

    switch (ConstExpr->getOpcode()) {
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
                                            DstType);
    } break;

    default:
      rc_return false;
    }

  } else {
    rc_return false;
  }

  rc_return true;
}

StringToken CCodeGenerator::handleSpecialFunction(const llvm::CallInst *Call) {

  auto *CalledFunc = Call->getCalledFunction();
  revng_assert(CalledFunc, "Special functions should all have a name");
  const auto &FuncName = CalledFunc->getName();

  StringToken Expression;

  if (FunctionTags::ModelGEP.isTagOf(CalledFunc)) {
    revng_assert(Call->getNumArgOperands() >= 2);

    // First argument is a string containing the base type
    auto *CurArg = Call->arg_begin();
    StringRef BaseTypeString = extractFromConstantStringPtr(CurArg->get());

    QualifiedType CurType = parseQualifiedType(BaseTypeString, Model);
    QualifiedType CurTypePtr = CurType;
    addPointerQualifier(CurTypePtr, Model);

    // Second argument is the base llvm::Value
    ++CurArg;
    llvm::Value *BaseValue = CurArg->get();

    Expression = buildCastExpr(TokenMap.at(BaseValue),
                               TypeMap.at(BaseValue),
                               CurTypePtr);
    ++CurArg;
    if (CurArg == Call->arg_end()) {
      // If there are no further arguments, we are casting from one reference
      // type to another
      Expression = buildDerefExpr(Expression);

    } else {
      // The base type is implicitly a pointer, so the first dereference should
      // use "->"
      StringToken CurExpr = addParentheses(Expression);
      StringRef DerefSymbol = "->";

      // Traverse the model to decide when to emit ".","->" or "[]"
      for (; CurArg != Call->arg_end(); ++CurArg) {
        flattenTypedefs(CurType);
        keepOnlyPtrAndArrayQualifiers(CurType);

        model::Qualifier *MainQualifier = nullptr;
        if (CurType.Qualifiers.size() > 0)
          MainQualifier = &CurType.Qualifiers.back();

        // If it's an array, add "[]"
        if (MainQualifier and model::Qualifier::isArray(*MainQualifier)) {
          CurExpr += "[";
          StringToken IndexExpr("");

          if (auto *Const = dyn_cast<llvm::ConstantInt>(CurArg->get())) {
            Const->getValue().toString(IndexExpr,
                                       /*base=*/10,
                                       /*signed=*/false);
          } else {
            IndexExpr = TokenMap.at(CurArg->get());
          }

          CurExpr += IndexExpr;
          CurExpr += "]";
          // Remove the qualifier we just analysed
          CurType.Qualifiers.pop_back();
          DerefSymbol = ".";
          continue;
        }

        // We shouldn't be going past pointers in a single ModelGEP
        revng_assert(not MainQualifier);
        CurExpr += DerefSymbol;
        auto *FieldIdxConst = cast<llvm::ConstantInt>(CurArg->get());
        uint64_t FieldIdx = FieldIdxConst->getValue().getLimitedValue();

        // Find the field name
        const auto *UnqualType = CurType.UnqualifiedType.getConst();

        if (auto *Struct = dyn_cast<model::StructType>(UnqualType)) {
          CurExpr += Struct->Fields.at(FieldIdx).name();
          CurType = Struct->Fields.at(FieldIdx).Type;

        } else if (auto *Union = dyn_cast<model::UnionType>(UnqualType)) {
          CurExpr += Union->Fields.at(FieldIdx).name();
          CurType = Union->Fields.at(FieldIdx).Type;

        } else {
          revng_abort("Unexpected ModelGEP type found: ");
          CurType.dump();
        }

        DerefSymbol = ".";
      }

      Expression = CurExpr;
    }
  } else if (FunctionTags::AddressOf.isTagOf(CalledFunc)) {
    // Second argument is the value being addressed
    const llvm::Value *Arg = Call->getArgOperand(1);
    Expression = buildAddressExpr(TokenMap.at(Arg));
  } else if (FunctionTags::AssignmentMarker.isTagOf(CalledFunc)) {
    const llvm::Value *Arg = Call->getArgOperand(0);

    // If a local variable has already been declared for the first argument,
    // this marker is transparent
    if (AlreadyDeclared.contains(Arg)) {
      return TokenMap.at(Arg);
    }

    if (TopScopeVariables.contains(Call)) {
      // If an entry in the TokenMap exists for this value, we have already
      // declared it, hence we just need to emit an assignment it here.
      revng_log(Log, "Already declared! Assigning " << Expression.str());
      Out << buildAssignmentExpr(TypeMap.at(Call),
                                 TokenMap.at(Call),
                                 TypeMap.at(Arg),
                                 TokenMap.at(Arg),
                                 /*WithDeclaration=*/false)
          << ";\n";
      Expression = TokenMap.at(Call);
    } else {

      revng_log(Log, "\tAssigning " << Expression.str());
      const StringToken VarName = NameGenerator.nextVarName(Call);
      revng_log(Log, "Declaring new local var for " << Expression.str());
      Out << buildAssignmentExpr(TypeMap.at(Call),
                                 VarName,
                                 TypeMap.at(Arg),
                                 TokenMap.at(Arg),
                                 /*WithDeclaration=*/true)
          << ";\n";
      Expression = VarName;
    }
  } else if (FunctionTags::StructInitializer.isTagOf(CalledFunc)) {
    // Struct initializers should be used only to pack together return values
    // of RawFunctionTypes that return multiple values, therefore they must have
    // the same type as the function's return type
    llvm::StructType *StructTy = cast<llvm::StructType>(Call->getType());
    revng_assert(Call->getFunction()->getReturnType() == StructTy);

    // Get the top function's prototype, which we can use to derive the type of
    // each field that is being initialized.
    auto *RawPrototype = cast<model::RawFunctionType>(&ParentPrototype);
    revng_assert(RawPrototype);

    // Emit LHS
    StringToken StructTyName = getReturnTypeName(*RawPrototype);

    // Declare a new variable that contains the struct
    const auto &VarName = NameGenerator.nextVarName();
    Out << StructTyName << " " << VarName << " = ";
    Expression = VarName;

    // Emit RHS
    char Separator = '{';
    for (const auto &[Arg, ArgType] :
         llvm::zip(Call->args(), RawPrototype->ReturnValues)) {

      Out << Separator << " "
          << buildCastExpr(TokenMap.at(Arg), TypeMap.at(Arg), ArgType.Type);
      Separator = ',';
    }
    Out << "};\n";
  } else if (FuncName.startswith("revng_stack_frame")) {

    // This expression has a pointer type associated to it because it represents
    // a pointer to a local variable, but the variable itself is not a pointer:
    // drop the qualifier when emitting the declaration
    Out << getNamedCInstance(dropPointer(TypeMap.at(Call)), StackFrameVarName)
        << ";\n";
    Expression = ("&" + StringRef(StackFrameVarName)).str();

  } else if (FuncName.startswith("revng_call_stack_arguments")) {
    StringToken VarName = NameGenerator.nextStackArgsVar();
    // This expression has a pointer type associated to it, but the actual
    // declaration that is emitted should not have such pointer.
    Out << getNamedCInstance(dropPointer(TypeMap.at(Call)), VarName) << ";\n ";
    Expression = ("&" + StringRef(VarName)).str();

  } else if (FuncName.startswith("revng_init_local_sp")) {
    // Note: we treat `revng_init_local_sp` separately from other `init`
    // functions so that we can assign a special name to the variable
    // representing the stack pointer.

    // Print the stack variable declaration
    Out << getNamedCInstance(TypeMap.at(Call), StackPtrVarName) << " = "
        << addParentheses(getNamedCInstance(TypeMap.at(Call), ""))
        << " revng_init_local_sp();\n";

    Expression = StackPtrVarName;

  } else if (FunctionTags::QEMU.isTagOf(CalledFunc)
             or FunctionTags::Helper.isTagOf(CalledFunc)
             or FuncName.startswith("llvm.") or FuncName.startswith("init_")) {

    Expression = buildFuncCallExpr(Call,
                                   model::Identifier::fromString(FuncName),
                                   /*prototype=*/nullptr);

    // If this call returns an aggregate type, we have to serialize the call
    // immediately. This is needed because the name of the type returned by this
    // function is not in the model: its name is derived from the called
    // function. If we wait for `AssignmentMarker` to emit a declaration for
    // it, we will loose information on which is the type of the returned
    // struct.
    if (Call->getType()->isAggregateType()) {
      StringToken VarName = NameGenerator.nextVarName(Call);
      Out << getReturnType(Call->getCalledFunction()) << " " << VarName << " = "
          << Expression << ";\n";

      AlreadyDeclared.insert(Call);
      Expression = VarName;
    }
  } else {
    revng_abort("Unknown non-isolated function");
  }

  return Expression;
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

  StringToken Expression;

  if (auto *Call = dyn_cast<llvm::CallInst>(&I)) {
    if (FunctionTags::CallToLifted.isTagOf(Call)) {
      revng_log(Log, "Emitting call to isolated function");

      // Retrieve the CallEdge
      const auto &[CallEdge, _] = getCallEdge(Model, Call);
      revng_assert(CallEdge);
      const auto &PrototypePath = getCallSitePrototype(Model, Call);

      // Construct the callee token (can be a function name or a function
      // pointer)
      StringToken CalleeToken;

      if (not isa<llvm::Function>(Call->getCalledOperand())) {
        // Indirect Call
        llvm::Value *CalledVal = Call->getCalledOperand();
        revng_assert(CalledVal);
        addOperandToken(CalledVal);
        auto &VarName = TokenMap.at(CalledVal);

        QualifiedType FuncPtrType = createPointerTo(PrototypePath, Model);

        // Cast the variable that holds the function pointer to the correct
        // type, and use it as the callee part of the expression
        CalleeToken = (addParentheses(getTypeName(FuncPtrType)) + VarName)
                        .str();
        CalleeToken = addParentheses(buildDerefExpr(CalleeToken));

      } else {
        if (not CallEdge->DynamicFunction.empty()) {
          // Dynamic Function
          auto &DynFuncID = CallEdge->DynamicFunction;
          auto &DynamicFunc = Model.ImportedDynamicFunctions.at(DynFuncID);
          CalleeToken = DynamicFunc.name();

        } else {
          // Isolated function
          llvm::Function *CalledFunc = Call->getCalledFunction();
          revng_assert(CalledFunc);
          const model::Function *ModelFunc = llvmToModelFunction(Model,
                                                                 *CalledFunc);
          revng_assert(ModelFunc);
          CalleeToken = ModelFunc->name();
        }
      }

      // Build the call expression
      revng_assert(not CalleeToken.empty());
      auto *Prototype = PrototypePath.get();
      Expression = buildFuncCallExpr(Call, CalleeToken, Prototype);

      if (auto *RawPrototype = dyn_cast<RawFunctionType>(Prototype)) {
        // RawFunctionTypes are allowed to return more than one value. In this
        // case, we need to derive the name of the struct that will hold the
        // returned values from the callee function, and emit the call
        // immediately.
        // If we were to postpone the emission to the next
        // `AssignmentMarker`, we would loose information on the name of the
        // return struct.
        if (Call->getType()->isAggregateType()) {
          StringToken VarName = NameGenerator.nextVarName(Call);
          Out << getReturnTypeName(*RawPrototype) << " " << VarName << " = "
              << Expression << ";\n";

          AlreadyDeclared.insert(Call);
          Expression = VarName;
        }
      } else if (auto *CPrototype = dyn_cast<CABIFunctionType>(Prototype)) {
        // CABIFunctionTypes are allowed to return arrays, which get enclosed in
        // a wrapper whose type name is derive by the callee. If we were to
        // postpone the emission to the next `AssignmentMarker`, we would
        // loose information on the name of the return struct.
        if (CPrototype->ReturnType.isArray()) {
          StringToken VarName = NameGenerator.nextVarName(Call);
          Out << getReturnTypeName(*CPrototype) << " " << VarName << " = "
              << Expression << ";\n";

          AlreadyDeclared.insert(Call);
          Expression = VarName;
        }
      } else {
        revng_abort("The called function has an unknown prototype type.");
      }
    } else {
      // Non-lifted function calls have to be dealt with separately
      Expression = handleSpecialFunction(Call);
    }

  } else if (auto *Load = dyn_cast<llvm::LoadInst>(&I)) {
    const llvm::Value *LoadedArg = Load->getPointerOperand();
    // The pointer operand's type and the actual loaded value's type might have
    // a mismatch. In this case, we want to cast the pointer operand to correct
    // type pointer before dereferencing it.
    QualifiedType ResultPtrType = TypeMap.at(Load);
    addPointerQualifier(ResultPtrType, Model);

    Expression = (buildDerefExpr(buildCastExpr(TokenMap.at(LoadedArg),
                                               TypeMap.at(LoadedArg),
                                               ResultPtrType)))
                   .str();

  } else if (auto *Store = dyn_cast<llvm::StoreInst>(&I)) {

    const llvm::Value *PointerOp = Store->getPointerOperand();
    const llvm::Value *ValueOp = Store->getValueOperand();
    // The LHS side of a store has the same type of the object pointed by the
    // pointer operand
    const QualifiedType PointedType = dropPointer(TypeMap.at(PointerOp));
    const QualifiedType &StoredType = TypeMap.at(ValueOp);

    if (StoredType.UnqualifiedType.isValid() and StoredType.isScalar()
        and not PointedType.is(model::TypeKind::PrimitiveType)) {
      // If we are storing a scalar value into a pointer to a struct, union or
      // pointer, cast the LHS to the scalar type before assigning it
      QualifiedType StoredPtrType = StoredType;
      addPointerQualifier(StoredPtrType, Model);
      StringToken CastedToken = buildCastExpr(TokenMap.at(PointerOp),
                                              TypeMap.at(PointerOp),
                                              StoredPtrType);
      Expression = buildAssignmentExpr(StoredType,
                                       buildDerefExpr(CastedToken),
                                       StoredType,
                                       TokenMap.at(ValueOp),
                                       /*WithDeclaration=*/false);
    } else {
      Expression = buildAssignmentExpr(PointedType,
                                       buildDerefExpr(TokenMap.at(PointerOp)),
                                       StoredType,
                                       TokenMap.at(ValueOp),
                                       /*WithDeclaration=*/false);
    }

  } else if (auto *Select = dyn_cast<llvm::SelectInst>(&I)) {

    StringToken Condition = TokenMap.at(Select->getCondition());
    const llvm::Value *Op1 = Select->getOperand(1);
    const llvm::Value *Op2 = Select->getOperand(1);

    StringToken Op1Token = buildCastExpr(TokenMap.at(Op1),
                                         TypeMap.at(Op1),
                                         TypeMap.at(Select));
    StringToken Op2Token = buildCastExpr(TokenMap.at(Op2),
                                         TypeMap.at(Op2),
                                         TypeMap.at(Select));

    Expression = (Condition + " ? " + addParentheses(Op1Token) + " : "
                  + addParentheses(Op2Token))
                   .str();

  } else if (auto *Alloca = dyn_cast<llvm::AllocaInst>(&I)) {

    // In LLVM IR, an alloca instruction returns a pointer, so the model type
    // associated to this value is actually a pointer to the model type of
    // the variable being allocated. Hence, to get the actual type of the
    // allocated variable, we must drop the pointer qualifier.
    const auto &AllocaType = TypeMap.at(Alloca);
    revng_assert(AllocaType.isPointer());
    const QualifiedType VarType = dropPointer(AllocaType);

    // Declare a local variable
    const StringToken VarName = NameGenerator.nextVarName(&I);
    Out << getNamedCInstance(VarType, VarName) << ";\n";
    // Use the address of this variable as the token associated to the alloca
    Expression = ("&" + VarName).str();

  } else if (auto *Ret = dyn_cast<llvm::ReturnInst>(&I)) {

    Expression = "return";

    if (llvm::Value *ReturnedVal = Ret->getReturnValue())
      Expression += (" " + TokenMap.at(ReturnedVal)).str();

  } else if (auto *Branch = dyn_cast<llvm::BranchInst>(&I)) {
    // This is never emitted directly in the BB: it is used when
    // emitting control-flow statements during the GHAST visit.
  } else if (auto *Switch = dyn_cast<llvm::SwitchInst>(&I)) {
    // This is never emitted directly in the BB: it is used when emitting
    // control-flow statements during the GHAST visit
  } else if (auto *IntToPtr = dyn_cast<llvm::IntToPtrInst>(&I)) {

    const llvm::Value *Operand = IntToPtr->getOperand(0);
    const QualifiedType &OpType = TypeMap.at(Operand);

    if (OpType.isPointer()) {
      // If the operand has already a pointer type in the model, IntToPtr has
      // no effect
      Expression = TokenMap.at(Operand);
    } else {
      // If we were not able to identify this value as a pointer, fallback to
      // converting directly its LLVM type to a QualifiedType
      QualifiedType PtrType = llvmIntToModelType(IntToPtr->getDestTy(), Model);
      Expression = buildCastExpr(TokenMap.at(Operand), OpType, PtrType);
    }
  } else if (auto *Bin = dyn_cast<llvm::BinaryOperator>(&I)) {

    const llvm::Value *Op1 = Bin->getOperand(0);
    const llvm::Value *Op2 = Bin->getOperand(1);
    const QualifiedType &ResultType = TypeMap.at(Bin);

    // TODO: Integer promotion
    Expression = (buildCastExpr(TokenMap.at(Op1), TypeMap.at(Op1), ResultType)
                  + getBinOpString(Bin)
                  + buildCastExpr(TokenMap.at(Op2),
                                  TypeMap.at(Op2),
                                  ResultType))
                   .str();
    Expression = addParentheses(Expression);

  } else if (auto *Cmp = dyn_cast<llvm::CmpInst>(&I)) {

    const llvm::Value *Op1 = Cmp->getOperand(0);
    const llvm::Value *Op2 = Cmp->getOperand(1);
    const QualifiedType &ResultType = llvmIntToModelType(Op1->getType(), Model);

    // TODO: Integer promotion
    Expression = (buildCastExpr(TokenMap.at(Op1), TypeMap.at(Op1), ResultType)
                  + getCmpOpString(Cmp->getPredicate())
                  + buildCastExpr(TokenMap.at(Op2),
                                  TypeMap.at(Op2),
                                  ResultType))
                   .str();
    Expression = addParentheses(Expression);

  } else if (auto *Cast = dyn_cast<llvm::CastInst>(&I)) {

    const llvm::Value *Op = Cast->getOperand(0);
    Expression = buildCastExpr(TokenMap.at(Op),
                               TypeMap.at(Op),
                               TypeMap.at(Cast));

  } else if (auto *ExtractVal = dyn_cast<llvm::ExtractValueInst>(&I)) {

    // Note: ExtractValues at this point should have been already
    // handled when visiting the instruction that generated their
    // struct operand
    // revng_assert(TokenMap.contains(&I));
    revng_assert(ExtractVal->getNumIndices() == 1);

    const auto &Idx = ExtractVal->getIndices().back();
    const llvm::Value *AggregateOp = ExtractVal->getAggregateOperand();
    const auto *AggregateType = cast<llvm::StructType>(AggregateOp->getType());

    Expression = (TokenMap.at(AggregateOp) + "."
                  + getFieldName(AggregateType, Idx).FieldName)
                   .str();

  } else if (auto *Unreach = dyn_cast<llvm::UnreachableInst>(&I)) {
    Expression = "__builtin_unreachable()";

  } else {
    revng_abort("Unexpected instruction found when decompiling");
  }

  return Expression;
}

void CCodeGenerator::emitBasicBlock(const llvm::BasicBlock *BB) {
  for (const Instruction &I : *BB) {

    // Guard this checking logger to prevent computing dumpToString if loggers
    // are not enabled.
    if (Log.isEnabled() or InlineLog.isEnabled())
      decompilerLog(Out, "Analyzing: " + dumpToString(I));

    StringToken Expression = buildExpression(I);

    if (not Expression.empty()) {
      if (I.getType()->isVoidTy()) {
        decompilerLog(Out, "Void instruction found: serializing expression");
        Out << Expression << ";\n";
      } else {
        decompilerLog(Out,
                      "Adding expression to the TokenMap " + Expression.str());
        TokenMap[&I] = Expression;
      }
    } else {
      decompilerLog(Out, "Nothing to serialize");
    }
  }
}

RecursiveCoroutine<StringToken>
CCodeGenerator::buildGHASTCondition(const ExprNode *E) {
  using NodeKind = ExprNode::NodeKind;
  switch (E->getKind()) {

  case NodeKind::NK_Atomic: {
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
    rc_return TokenMap.at(Br->getCondition());
  } break;

  case NodeKind::NK_Not: {
    const NotNode *N = cast<NotNode>(E);
    ExprNode *Negated = N->getNegatedNode();

    StringToken Expression;
    Expression = ("!" + addParentheses(rc_recur buildGHASTCondition(Negated)))
                   .str();
    rc_return Expression;
  } break;

  case NodeKind::NK_And:
  case NodeKind::NK_Or: {
    const BinaryNode *Binary = cast<BinaryNode>(E);

    const auto &[Child1, Child2] = Binary->getInternalNodes();
    const auto Child1Token = rc_recur buildGHASTCondition(Child1);
    const auto Child2Token = rc_recur buildGHASTCondition(Child2);
    const llvm::StringRef OpToken = E->getKind() == NodeKind::NK_And ? " && " :
                                                                       " || ";
    StringToken Expression = addParentheses(Child1Token);
    Expression += OpToken;
    Expression += addParentheses(Child2Token);
    rc_return Expression;
  } break;

  default:
    revng_abort("Unknown ExprNode kind");
  }
}

RecursiveCoroutine<void> CCodeGenerator::emitGHASTNode(const ASTNode *N) {
  if (N == nullptr)
    rc_return;

  auto Kind = N->getKind();
  switch (Kind) {

  case ASTNode::NodeKind::NK_Break: {
    const BreakNode *Break = llvm::cast<BreakNode>(N);
    if (Break->breaksFromWithinSwitch()) {
      revng_assert(not SwitchStateVars.empty()
                   and not SwitchStateVars.back().empty());
      Out << SwitchStateVars.back() << " = true;\n";
    }
  };
    [[fallthrough]];

  case ASTNode::NodeKind::NK_SwitchBreak: {
    Out << "break;\n";
  } break;

  case ASTNode::NodeKind::NK_Continue: {
    const ContinueNode *Continue = cast<ContinueNode>(N);

    // Print the condition computation code of the if statement.
    if (Continue->hasComputation()) {
      IfNode *ComputationIfNode = Continue->getComputationIfNode();
      buildGHASTCondition(ComputationIfNode->getCondExpr());

      // Actually print the continue statement only if the continue is not
      // implicit (i.e. it is not the last statement of the loop).
      if (not Continue->isImplicit())
        Out << "continue;\n";
    }
  } break;

  case ASTNode::NodeKind::NK_Code: {
    const CodeNode *Code = cast<CodeNode>(N);
    llvm::BasicBlock *BB = Code->getOriginalBB();
    revng_assert(BB != nullptr);
    emitBasicBlock(BB);
  } break;

  case ASTNode::NodeKind::NK_If: {
    const IfNode *If = cast<IfNode>(N);
    const StringToken CondExpr = buildGHASTCondition(If->getCondExpr());
    // "If" expression
    // TODO: possibly cast the CondExpr if it's not convertible to boolean?
    revng_assert(not CondExpr.empty());
    Out << "if (" + CondExpr + ") {\n";

    // "Then" expression (always emitted)
    if (nullptr == If->getThen()) {
      Out << " // Empty\n";
    } else {
      rc_recur emitGHASTNode(If->getThen());
    }
    Out << "}\n";

    // "Else" expression (optional)
    if (If->hasElse()) {
      Out << "else {\n";
      rc_recur emitGHASTNode(If->getElse());
      Out << "}\n";
    }

    break;
  } break;

  case ASTNode::NodeKind::NK_Scs: {
    const ScsNode *LoopBody = cast<ScsNode>(N);

    // Calculate the string of the condition
    // TODO: possibly cast the CondExpr if it's not convertible to boolean?
    StringToken CondExpr("true");
    if (LoopBody->isDoWhile() or LoopBody->isWhile()) {
      const IfNode *LoopCondition = LoopBody->getRelatedCondition();
      revng_assert(LoopCondition);

      // Retrieve the expression of the condition as well as emitting its
      // associated basic block
      CondExpr = buildGHASTCondition(LoopCondition->getCondExpr());
      revng_assert(not CondExpr.empty());
    }

    if (LoopBody->isDoWhile())
      Out << "do ";
    else
      Out << "while (" + CondExpr + ") ";

    revng_assert(LoopBody->hasBody());
    Out << "{\n";
    rc_recur emitGHASTNode(LoopBody->getBody());
    Out << "}";

    if (LoopBody->isDoWhile())
      Out << " while (" + CondExpr + ");\n";

  } break;

  case ASTNode::NodeKind::NK_List: {
    const SequenceNode *Seq = cast<SequenceNode>(N);
    for (const ASTNode *Child : Seq->nodes())
      rc_recur emitGHASTNode(Child);

  } break;

  case ASTNode::NodeKind::NK_Switch: {
    const SwitchNode *Switch = cast<SwitchNode>(N);

    // If needed, print the declaration of the switch state variable, which
    // is used by nested switches inside loops to break out of the loop
    if (Switch->needsStateVariable()) {
      revng_assert(Switch->needsLoopBreakDispatcher());
      SwitchStateVars.push_back(NameGenerator.nextSwitchStateVar());
      Out << "bool " << SwitchStateVars.back() << " = false;\n";
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
      revng_assert(not LoopStateVar.empty());
      // This switch does not come from an instruction: it's a dispatcher
      // for the loop state variable
      SwitchVarToken = LoopStateVar;

      // TODO: finer decision on the type of the loop state variable
      using model::PrimitiveTypeKind::Unsigned;
      SwitchVarType.UnqualifiedType = Model.getPrimitiveType(Unsigned, 8);
    }
    revng_assert(not SwitchVarToken.empty());

    if (not SwitchVarType.is(model::TypeKind::PrimitiveType)) {
      model::QualifiedType BoolTy;
      // TODO: finer decision on how to cast structs used in a switch
      using model::PrimitiveTypeKind::Unsigned;
      BoolTy.UnqualifiedType = Model.getPrimitiveType(Unsigned, 8);

      SwitchVarToken = buildCastExpr(SwitchVarToken, SwitchVarType, BoolTy);
    }

    // Generate the switch statement
    Out << "switch (" << SwitchVarToken << ") {\n";

    // Generate the body of the switch (except for the default)
    for (const auto &[Labels, CaseNode] : Switch->cases_const_range()) {
      revng_assert(not Labels.empty());
      // Generate the case label(s) (multiple case labels might share the
      // same body)
      for (uint64_t CaseVal : Labels) {
        Out << "case ";
        if (SwitchVar) {
          llvm::Type *SwitchVarT = SwitchVar->getType();
          auto *IntType = cast<llvm::IntegerType>(SwitchVarT);
          auto *CaseConst = llvm::ConstantInt::get(IntType, CaseVal);
          // TODO: assigned the signedness based on the signedness of the
          // condition
          CaseConst->getValue().print(Out, false);
        } else {
          Out << CaseVal;
        }
        Out << ":\n";
      }
      Out << "{\n";

      // Generate the case body
      rc_recur emitGHASTNode(CaseNode);

      Out << "} break;\n";
    }

    // Generate the default case if it exists
    if (auto *Default = Switch->getDefault()) {
      Out << "default: {\n";
      rc_recur emitGHASTNode(Default);
      Out << "} break;\n";
    }

    Out << "}\n";

    // If the switch needs a loop break dispatcher, reset the associated
    // state variable before emitting the switch statement.
    if (Switch->needsLoopBreakDispatcher()) {
      revng_assert(not SwitchStateVars.empty()
                   and not SwitchStateVars.back().empty());
      Out << "if (" << SwitchStateVars.back() << ")\nbreak;\n";
    }

    // If we're done with a switch that generates a state variable to break out
    // of loops, pop it from the stack.
    if (Switch->needsStateVariable()) {
      revng_assert(Switch->needsLoopBreakDispatcher());
      SwitchStateVars.pop_back();
    }

  } break;

  case ASTNode::NodeKind::NK_Set: {
    const SetNode *Set = cast<SetNode>(N);
    unsigned StateValue = Set->getStateVariableValue();
    revng_assert(not LoopStateVar.empty());

    // Print an assignment to the loop state variable. This is an artificial
    // variable introduced by the GHAST to enable executing certain pieces
    // of code based on which control-flow branch was taken. This, for
    // example, can be used to jump to the middle of a loop
    // instead of at the start, without emitting gotos.
    Out << LoopStateVar << " = " << StateValue << ";\n";
  } break;
  }

  rc_return;
}

void CCodeGenerator::emitFunction(bool NeedsLocalStateVar) {

  // Create a token for each of the function's arguments
  if (auto *RawPrototype = dyn_cast<model::RawFunctionType>(&ParentPrototype)) {

    const auto &ModelArgs = RawPrototype->Arguments;
    const auto &LLVMArgs = LLVMFunction.args();
    const auto LLVMArgsNum = LLVMFunction.arg_size();

    const auto &StackArgType = RawPrototype->StackArgumentsType.UnqualifiedType;
    const auto ArgSize = ModelArgs.size();
    revng_assert(LLVMArgsNum == ArgSize
                 or (LLVMArgsNum == ArgSize + 1 and StackArgType.isValid()));

    // Associate each LLVM argument with its name
    for (const auto &[ModelArg, LLVMArg] :
         llvm::zip_first(ModelArgs, LLVMArgs)) {
      revng_log(Log, "Adding token for: " << dumpToString(LLVMArg));
      TokenMap[&LLVMArg] = ModelArg.name().str();
    }

    // Add a token for the stack arguments
    if (StackArgType.isValid()) {
      const auto *LLVMArg = LLVMFunction.getArg(LLVMArgsNum - 1);
      revng_log(Log, "Adding token for: " << dumpToString(LLVMArg));
      TokenMap[LLVMArg] = "stack_args";
    }
  } else if (auto *CPrototype = dyn_cast<CABIFunctionType>(&ParentPrototype)) {

    const auto &ModelArgs = CPrototype->Arguments;
    const auto &LLVMArgs = LLVMFunction.args();

    revng_assert(LLVMFunction.arg_size() == ModelArgs.size());

    // Associate each LLVM argument with its name
    for (const auto &[ModelArg, LLVMArg] : llvm::zip(ModelArgs, LLVMArgs)) {
      revng_log(Log, "Adding token for: " << dumpToString(LLVMArg));
      TokenMap[&LLVMArg] = ModelArg.name().str();
    }
  } else {
    revng_abort("Functions can only have RawFunctionType or "
                "CABIFunctionType.");
  }

  // Print function's prototype
  printFunctionPrototype(ParentPrototype, ModelFunction.name(), Out, Model);
  Out << " {\n";

  // Emit a declaration for the loop state variable, which is used to
  // redirect control flow inside loops (e.g. if we want to jump in the
  // middle of a loop during a certain iteration)
  if (NeedsLocalStateVar)
    Out << "uint64_t " << LoopStateVar << ";\n";

  // Declare all variables that have the entire function as a scope
  for (const llvm::Value *VarToDeclare : TopScopeVariables) {
    auto VarName = NameGenerator.nextVarName();
    auto VarType = TypeMap.at(VarToDeclare);
    Out << getNamedCInstance(VarType, VarName) << ";\n";
    TokenMap[VarToDeclare] = VarName;
  }

  // Recursively print the body of this function
  emitGHASTNode(GHAST.getRoot());

  Out << "}\n";
}

std::string decompileFunction(const llvm::Function &LLVMFunc,
                              const ASTTree &CombedAST,
                              const Binary &Model,
                              const ValueSet &TopScopeVariables,
                              bool NeedsLocalStateVar) {
  std::string Result;

  llvm::raw_string_ostream Out(Result);
  CCodeGenerator Backend(Model, LLVMFunc, CombedAST, TopScopeVariables, Out);
  Backend.emitFunction(NeedsLocalStateVar);
  Out.flush();

  return Result;
}

void decompile(llvm::Module &Module,
               const model::Binary &Model,
               revng::pipes::FunctionStringMap &DecompiledFunctions) {

  if (Log.isEnabled())
    writeToFile(Model.toString(), "model-during-c-codegen.yaml");

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
    auto TopScopeVariables = collectLocalVariables(F);
    auto NeedsLoopStateVar = hasLoopDispatchers(GHAST);
    std::string CCode = decompileFunction(F,
                                          GHAST,
                                          Model,
                                          TopScopeVariables,
                                          NeedsLoopStateVar);

    // Push the C code into
    MetaAddress Key = getMetaAddressMetadata(&F, "revng.function.entry");
    DecompiledFunctions.insert_or_assign(Key, std::move(CCode));
  }
}
