//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <variant>

#include "llvm/ADT/ScopeExit.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/PTMLC.h"
#include "revng/TypeNames/PTMLCTypeBuilder.h"
#include "revng/mlir/Dialect/Clift/Utils/CBackend.h"

namespace clift = mlir::clift;

using namespace mlir::clift;

namespace {

static RecursiveCoroutine<void> noopCoroutine() {
  rc_return;
}

template<typename Operation = mlir::Operation *>
static Operation getOnlyOperation(mlir::Region &R) {
  revng_assert(R.hasOneBlock());
  mlir::Block &B = R.front();
  auto Beg = B.begin();
  auto End = B.end();

  if (Beg == End)
    return {};

  mlir::Operation *Op = &*Beg;

  if (++Beg != End)
    return {};

  if constexpr (std::is_same_v<Operation, mlir::Operation *>) {
    return Op;
  } else {
    return mlir::dyn_cast<Operation>(Op);
  }
}

static llvm::StringRef getCIntegerLiteralSuffix(const CIntegerKind Integer,
                                                const bool Signed) {
  switch (Integer) {
  default:
  case CIntegerKind::Int:
    return Signed ? "" : "u";
  case CIntegerKind::Long:
    return Signed ? "l" : "ul";
  case CIntegerKind::LongLong:
    return Signed ? "ll" : "ull";
  }
}

using Keyword = ptml::CBuilder::Keyword;
using Operator = ptml::CBuilder::Operator;

enum class OperatorPrecedence {
  Parentheses,
  Comma,
  Assignment,
  Or,
  And,
  Bitor,
  Bitxor,
  Bitand,
  Equality,
  Relational,
  Shift,
  Additive,
  Multiplicative,
  UnaryPrefix,
  UnaryPostfix,
  Primary,
};

class CEmitter {
public:
  explicit CEmitter(const TargetCImplementation &Target,
                    ptml::CTypeBuilder &Builder,
                    llvm::raw_ostream &Out) :
    Target(Target), C(Builder), Out(Out, C) {
    Builder.setOutputStream(this->Out);
  }

  const model::Segment &getModelSegment(GlobalVariableOp Op) {
    auto L = pipeline::locationFromString(revng::ranks::Segment,
                                          Op.getUniqueHandle());
    if (not L)
      revng_abort("Unrecognizable global variable unique handle.");

    auto Key = L->at(revng::ranks::Segment);
    auto It = C.Binary.Segments().find(Key);
    if (It == C.Binary.Segments().end())
      revng_abort("No matching model segment.");
    return *It;
  }

  using ModelFunctionVariant = std::variant<const model::Function *,
                                            const model::DynamicFunction *>;

  ModelFunctionVariant getModelFunctionVariant(FunctionOp Op) {
    if (auto L = pipeline::locationFromString(revng::ranks::Function,
                                              Op.getUniqueHandle())) {
      auto [Key] = L->at(revng::ranks::Function);
      auto It = C.Binary.Functions().find(Key);
      if (It == C.Binary.Functions().end())
        revng_abort("No matching model function.");
      return &*It;
    }

    if (auto L = pipeline::locationFromString(revng::ranks::DynamicFunction,
                                              Op.getUniqueHandle())) {
      auto [Key] = L->at(revng::ranks::DynamicFunction);
      auto It = C.Binary.ImportedDynamicFunctions().find(Key);
      if (It == C.Binary.ImportedDynamicFunctions().end())
        revng_abort("No matching model dynamic function.");
      return &*It;
    }

    revng_abort("Unrecognizable function unique handle.");
  }

  const model::Function &getModelFunction(FunctionOp Op) {
    auto Variant = getModelFunctionVariant(Op);
    if (auto F = std::get_if<const model::Function *>(&Variant))
      return **F;
    revng_abort("Expected isolated model function.");
  }

  std::optional<uint64_t>
  parseModelTypeUniqueHandle(llvm::StringRef UniqueHandle) {
    if (not UniqueHandle.consume_front("/model-type/"))
      return std::nullopt;

    uint64_t ID;
    if (UniqueHandle.consumeInteger(/*Radix=*/10, ID))
      return std::nullopt;

    if (not UniqueHandle.empty())
      return std::nullopt;

    return ID;
  }

  const model::TypeDefinition *
  getModelTypeDefinition(uint64_t ID, model::TypeDefinitionKind::Values Kind) {
    auto It = C.Binary.TypeDefinitions().find({ ID, Kind });
    if (It == C.Binary.TypeDefinitions().end())
      return nullptr;
    return It->get();
  }

  const model::TypeDefinition &getModelTypeDefinition(TypeDefinitionAttr Type) {
    using TypeKind = model::TypeDefinitionKind::Values;

    auto MaybeID = parseModelTypeUniqueHandle(Type.getUniqueHandle());
    if (not MaybeID)
      revng_abort("Unrecognized type unique handle");

    if (mlir::isa<FunctionTypeAttr>(Type)) {
      auto CF = getModelTypeDefinition(*MaybeID,
                                       TypeKind::CABIFunctionDefinition);
      auto RF = getModelTypeDefinition(*MaybeID,
                                       TypeKind::RawFunctionDefinition);

      if (CF != nullptr and RF != nullptr)
        revng_abort("Ambiguous model function type definition");

      if (CF != nullptr)
        return *CF;

      if (RF != nullptr)
        return *RF;

      revng_abort("No matching model function type definition.");
    }

    model::TypeDefinitionKind::Values Kind;
    if (mlir::isa<TypedefTypeAttr>(Type))
      Kind = model::TypeDefinitionKind::TypedefDefinition;
    else if (mlir::isa<EnumTypeAttr>(Type))
      Kind = model::TypeDefinitionKind::EnumDefinition;
    else if (mlir::isa<StructTypeAttr>(Type))
      Kind = model::TypeDefinitionKind::StructDefinition;
    else if (mlir::isa<UnionTypeAttr>(Type))
      Kind = model::TypeDefinitionKind::UnionDefinition;
    else
      revng_abort("Unsupported type definition");

    auto ModelType = getModelTypeDefinition(*MaybeID, Kind);
    if (ModelType == nullptr)
      revng_abort("No matching model type definition.");
    return *ModelType;
  }

  void emitPrimitiveType(PrimitiveType Type) {
    auto Kind = static_cast<model::PrimitiveKind::Values>(Type.getKind());
    auto ModelType = model::PrimitiveType::make(Kind, Type.getSize());
    Out << C.getLocationReference(llvm::cast<model::PrimitiveType>(*ModelType));
  }

  RecursiveCoroutine<void>
  emitDeclaration(ValueType Type,
                  std::optional<llvm::StringRef> DeclaratorName) {
    // Function type expansion is currently always disabled:
    static constexpr bool ExpandFunctionTypes = false;

    enum class StackItemKind {
      Terminal,
      Pointer,
      Array,
      Function,
    };

    struct StackItem {
      StackItemKind Kind;
      ValueType Type;
    };

    llvm::SmallVector<StackItem> Stack;

    bool NeedSpace = false;
    auto EmitSpace = [&]() {
      if (NeedSpace)
        Out << ' ';
      NeedSpace = false;
    };

    auto EmitConst = [&](ValueType T) {
      EmitSpace();
      if (T.isConst())
        Out << C.getKeyword(Keyword::Const) << ' ';
    };

    // Recurse through the declaration, pushing each level into the stack until
    // a terminal type is encountered. Primitive types as well as defined types
    // are considered terminal. Function types are not considered terminal if
    // function type expansion is enabled.
    while (true) {
      StackItem Item = { StackItemKind::Terminal, Type };

      if (auto T = mlir::dyn_cast<PrimitiveType>(Type)) {
        EmitConst(T);
        emitPrimitiveType(T);
        NeedSpace = true;
      } else if (auto T = mlir::dyn_cast<PointerType>(Type)) {
        Item.Kind = StackItemKind::Pointer;
        Type = T.getPointeeType();
      } else if (auto T = mlir::dyn_cast<ArrayType>(Type)) {
        Item.Kind = StackItemKind::Array;
        Type = T.getElementType();
      } else if (auto T = mlir::dyn_cast<DefinedType>(Type)) {
        auto D = T.getElementType();
        auto F = mlir::dyn_cast<FunctionTypeAttr>(D);

        // Expand the function type if function type expansion is enabled.
        if (F and ExpandFunctionTypes) {
          Item.Kind = StackItemKind::Function;
          Type = F.getReturnType();
        } else {
          if (mlir::isa<EnumTypeAttr>(D))
            Out << C.getKeyword(Keyword::Enum) << ' ';
          else if (mlir::isa<StructTypeAttr>(D))
            Out << C.getKeyword(Keyword::Struct) << ' ';
          else if (mlir::isa<UnionTypeAttr>(D))
            Out << C.getKeyword(Keyword::Union) << ' ';

          EmitConst(T);
          Out << C.getLocationReference(getModelTypeDefinition(D));
          NeedSpace = true;
        }
      }

      Stack.push_back(Item);

      if (Item.Kind == StackItemKind::Terminal)
        break;
    }

    // Print type syntax appearing before the declarator name. This includes
    // cv-qualifiers, stars indicating a pointer, as well as left parentheses
    // used to disambiguate non-root array and function types. The types must be
    // handled inside out, so the stack is visited in reverse order.
    for (auto [RI, SI] : llvm::enumerate(std::views::reverse(Stack))) {
      const size_t I = Stack.size() - RI - 1;

      switch (SI.Kind) {
      case StackItemKind::Terminal: {
        // Do nothing
      } break;
      case StackItemKind::Pointer: {
        auto T = mlir::dyn_cast<PointerType>(SI.Type);
        if (T.getPointerSize() != Target.PointerSize)
          revng_abort("Pointer is not representable on the target platform.");
        EmitSpace();
        Out << '*';
      } break;
      case StackItemKind::Array: {
        if (I != 0 and Stack[I - 1].Kind != StackItemKind::Array) {
          Out << '(';
          NeedSpace = false;
        }
      } break;
      case StackItemKind::Function: {
        if (I != 0) {
          Out << '(';
          NeedSpace = false;
        }
      } break;
      }

      if (SI.Kind != StackItemKind::Terminal)
        EmitConst(SI.Type);
    }

    if (DeclaratorName) {
      EmitSpace();
      Out << *DeclaratorName;
    }

    // Print type syntax appearing after the declarator name. This includes
    // right parentheses matching the left parentheses printed in the first
    // pass, as well as array extents and function parameter lists. The
    // declarators appearing in function parameter lists are printed by
    // recursively entering this function.
    for (auto [I, SI] : llvm::enumerate(Stack)) {
      switch (SI.Kind) {
      case StackItemKind::Terminal: {
        // Do nothing
      } break;
      case StackItemKind::Pointer: {
        // Do nothing
      } break;
      case StackItemKind::Array: {
        if (I != 0 and Stack[I - 1].Kind != StackItemKind::Array)
          Out << ')';

        Out << '[';
        Out << mlir::cast<ArrayType>(SI.Type).getElementsCount();
        Out << ']';
      } break;
      case StackItemKind::Function: {
        auto T = mlir::dyn_cast<DefinedType>(SI.Type);
        auto F = mlir::dyn_cast<FunctionTypeAttr>(T.getElementType());

        if (I != 0)
          Out << ')';

        Out << '(';
        if (F.getArgumentTypes().empty()) {
          Out << C.tokenTag("void", ptml::c::tokens::Type);
        } else {
          for (auto [J, PT] : llvm::enumerate(F.getArgumentTypes())) {
            if (J != 0)
              Out << ',' << ' ';

            rc_recur emitType(PT);
          }
        }
        Out << ')';
      } break;
      }
    }
  }

  RecursiveCoroutine<void> emitType(ValueType Type) {
    return emitDeclaration(Type, std::nullopt);
  }

  static OperatorPrecedence decrementPrecedence(OperatorPrecedence Precedence) {
    revng_assert(Precedence != static_cast<OperatorPrecedence>(0));
    using T = std::underlying_type_t<OperatorPrecedence>;
    return static_cast<OperatorPrecedence>(static_cast<T>(Precedence) - 1);
  }

  ptml::Tag
  getIntegerConstant(uint64_t Value, CIntegerKind Integer, bool Signed) {
    llvm::SmallString<64> String;
    {
      llvm::raw_svector_ostream Stream(String);

      if (Signed and static_cast<int64_t>(Value) < 0) {
        Stream << static_cast<int64_t>(Value);
      } else {
        Stream << Value;
      }

      Stream << getCIntegerLiteralSuffix(Integer, Signed);
    }
    return C.getConstantTag(String);
  }

  void emitIntegerImmediate(uint64_t Value, ValueType Type) {
    Type = dealias(Type, /*IgnoreQualifiers=*/true);

    if (auto T = mlir::dyn_cast<PrimitiveType>(Type)) {
      auto Integer = Target.getIntegerKind(T.getSize());

      if (not Integer) {
        // Emit explicit cast if the standard integer type is not known. Emit
        // the literal itself without a suffix (as if int).

        Out << '(';
        emitPrimitiveType(T);
        Out << ')';

        Integer = CIntegerKind::Int;
      }

      bool Signed = T.getKind() == PrimitiveKind::SignedKind;
      Out << getIntegerConstant(Value, *Integer, Signed);
    } else {
      auto TypeAttr = mlir::cast<DefinedType>(Type).getElementType();
      const auto &ModelType = getModelTypeDefinition(TypeAttr);
      const auto &ModelEnum = llvm::cast<model::EnumDefinition>(ModelType);

      auto It = ModelEnum.Entries().find(Value);
      if (It == ModelEnum.Entries().end())
        revng_abort("Model enum entry not found.");

      Out << C.getLocation(/*IsDefinition=*/false, ModelEnum, *It);
    }
  }

  llvm::StringRef getLocalSymbolName(mlir::Operation *Op,
                                     llvm::StringRef Prefix,
                                     size_t &Counter) {
    auto [Iterator, Inserted] = LocalSymbolNames.try_emplace(Op);
    if (Inserted) {
      std::string Symbol;

      while (true) {
        llvm::raw_string_ostream(Symbol) << '_' << Prefix << '_' << Counter++;

        if (not llvm::is_contained(ParameterNames, Symbol))
          break;

        Symbol.clear();
      }

      Iterator->second = std::move(Symbol);
    }
    return Iterator->second;
  }

  llvm::StringRef getLocalSymbolName(LocalVariableOp Op) {
    return getLocalSymbolName(Op.getOperation(), "var", LocalVariableCounter);
  }

  llvm::StringRef getLocalSymbolName(MakeLabelOp Op) {
    return getLocalSymbolName(Op.getOperation(), "label", GotoLabelCounter);
  }

  //===---------------------------- Expressions ---------------------------===//

  RecursiveCoroutine<void> emitImmediateExpression(mlir::Value V) {
    auto E = V.getDefiningOp<ImmediateOp>();
    emitIntegerImmediate(E.getValue(), E.getResult().getType());
    rc_return;
  }

  RecursiveCoroutine<void> emitParameterExpression(mlir::Value V) {
    auto Arg = mlir::cast<mlir::BlockArgument>(V);
    Out << ParameterNames[Arg.getArgNumber()];
    rc_return;
  }

  RecursiveCoroutine<void> emitLocalVariableExpression(mlir::Value V) {
    // TODO: Emit variable name from the model once the model is extended to
    //       provide this information.

    auto Symbol = getLocalSymbolName(V.getDefiningOp<LocalVariableOp>());
    Out << C.getVariableLocationReference(Symbol, *CurrentFunction);

    rc_return;
  }

  RecursiveCoroutine<void> emitUseExpression(mlir::Value V) {
    auto E = V.getDefiningOp<UseOp>();

    auto Module = E->getParentOfType<clift::ModuleOp>();
    revng_assert(Module);

    mlir::Operation
      *SymbolOp = mlir::SymbolTable::lookupSymbolIn(Module,
                                                    E.getSymbolNameAttr());
    revng_assert(SymbolOp);

    if (auto G = mlir::dyn_cast<GlobalVariableOp>(SymbolOp)) {
      Out << C.getLocationReference(getModelSegment(G));
    } else if (auto F = mlir::dyn_cast<FunctionOp>(SymbolOp)) {
      auto Visitor = [&](const auto *ModelFunction) {
        Out << C.getLocationReference(*ModelFunction);
      };
      std::visit(Visitor, getModelFunctionVariant(F));
    } else {
      revng_abort("Unsupported global operation");
    }

    rc_return;
  }

  template<typename Class>
  void emitClassMemberReference(const Class &TheClass, uint64_t Key) {
    auto It = TheClass.Fields().find(Key);
    if (It == TheClass.Fields().end())
      revng_abort("Class member not found.");

    Out << C.getLocation(/*IsDefinition=*/false, TheClass, *It);
  }

  RecursiveCoroutine<void> emitAccessExpression(mlir::Value V) {
    auto E = V.getDefiningOp<AccessOp>();

    // Parenthesizing a nested unary postfix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);

    rc_recur emitExpression(E.getValue());

    Out << C.getOperator(E.isIndirect() ? Operator::Arrow : Operator::Dot);

    const model::TypeDefinition
      &ModelType = getModelTypeDefinition(E.getClassTypeAttr());

    if (auto *T = llvm::dyn_cast<model::StructDefinition>(&ModelType))
      emitClassMemberReference(*T, E.getFieldAttr().getOffset());
    else if (auto *T = llvm::dyn_cast<model::UnionDefinition>(&ModelType))
      emitClassMemberReference(*T, E.getMemberIndex());
  }

  RecursiveCoroutine<void> emitSubscriptExpression(mlir::Value V) {
    auto E = V.getDefiningOp<SubscriptOp>();

    // Parenthesizing a nested unary postfix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);

    rc_recur emitExpression(E.getPointer());

    // The precedence here could be parentheses and still preserve semantics,
    // but given that a comma expression within a subscript ( array[i, j] ) is
    // not only very confusing, but has a different meaning in C++23, we force
    // comma expressions to be parenthesized, the same way they are in argument
    // lists. The output in this case is as: array[(i, j)]
    CurrentPrecedence = OperatorPrecedence::Comma;

    Out << '[';
    rc_recur emitExpression(E.getIndex());
    Out << ']';
  }

  RecursiveCoroutine<void> emitCallExpression(mlir::Value V) {
    auto E = V.getDefiningOp<CallOp>();

    // Parenthesizing a nested unary postfix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);

    rc_recur emitExpression(E.getFunction());

    // The precedence here must be comma, because an argument list cannot
    // contain an unparenthesized comma expression. It would be parsed as two
    // arguments instead.
    CurrentPrecedence = OperatorPrecedence::Comma;

    Out << '(';
    for (auto [I, A] : llvm::enumerate(E.getArguments())) {
      if (I != 0)
        Out << ',' << ' ';

      rc_recur emitExpression(A);
    }
    Out << ')';
  }

  RecursiveCoroutine<void> emitCastExpression(mlir::Value V) {
    auto E = V.getDefiningOp<CastOp>();

    if (E.getKind() != CastKind::Decay) {
      Out << '(';
      rc_recur emitType(E.getResult().getType());
      Out << ')';
    }

    // Parenthesizing a nested unary postfix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);

    rc_recur emitExpression(E.getValue());
  }

  static ptml::CBuilder::Operator getOperator(mlir::Operation *Op) {
    if (mlir::isa<NegOp>(Op))
      return Operator::UnaryMinus;
    if (mlir::isa<AddOp>(Op))
      return Operator::Add;
    if (mlir::isa<SubOp>(Op))
      return Operator::Sub;
    if (mlir::isa<MulOp>(Op))
      return Operator::Mul;
    if (mlir::isa<DivOp>(Op))
      return Operator::Div;
    if (mlir::isa<RemOp>(Op))
      return Operator::Modulo;
    if (mlir::isa<LogicalNotOp>(Op))
      return Operator::BoolNot;
    if (mlir::isa<LogicalAndOp>(Op))
      return Operator::BoolAnd;
    if (mlir::isa<LogicalOrOp>(Op))
      return Operator::BoolOr;
    if (mlir::isa<BitwiseNotOp>(Op))
      return Operator::BinaryNot;
    if (mlir::isa<BitwiseAndOp>(Op))
      return Operator::And;
    if (mlir::isa<BitwiseOrOp>(Op))
      return Operator::Or;
    if (mlir::isa<BitwiseXorOp>(Op))
      return Operator::Xor;
    if (mlir::isa<ShiftLeftOp>(Op))
      return Operator::LShift;
    if (mlir::isa<ShiftRightOp>(Op))
      return Operator::RShift;
    if (mlir::isa<EqualOp>(Op))
      return Operator::CmpEq;
    if (mlir::isa<NotEqualOp>(Op))
      return Operator::CmpNeq;
    if (mlir::isa<LessThanOp>(Op))
      return Operator::CmpLt;
    if (mlir::isa<GreaterThanOp>(Op))
      return Operator::CmpGt;
    if (mlir::isa<LessThanOrEqualOp>(Op))
      return Operator::CmpLte;
    if (mlir::isa<GreaterThanOrEqualOp>(Op))
      return Operator::CmpGte;
    if (mlir::isa<IncrementOp, PostIncrementOp>(Op))
      return Operator::Increment;
    if (mlir::isa<DecrementOp, PostDecrementOp>(Op))
      return Operator::Decrement;
    if (mlir::isa<AddressofOp>(Op))
      return Operator::AddressOf;
    if (mlir::isa<IndirectionOp>(Op))
      return Operator::PointerDereference;
    if (mlir::isa<AssignOp>(Op))
      return Operator::Assign;
    if (mlir::isa<CommaOp>(Op))
      return Operator::Comma;
    revng_abort("This operation does not represent a C operator.");
  }

  RecursiveCoroutine<void> emitPrefixExpression(mlir::Value V) {
    mlir::Operation *Op = V.getDefiningOp();
    Out << C.getOperator(getOperator(Op));

    // Parenthesizing a nested unary prefix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPrefix);

    return emitExpression(Op->getOperand(0));
  }

  RecursiveCoroutine<void> emitPostfixExpression(mlir::Value V) {
    mlir::Operation *Op = V.getDefiningOp();
    rc_recur emitExpression(Op->getOperand(0));

    // Parenthesizing a nested unary postfix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);

    Out << C.getOperator(getOperator(Op));
  }

  RecursiveCoroutine<void> emitInfixExpression(mlir::Value V) {
    mlir::Operation *Op = V.getDefiningOp();

    auto LhsPrecedence = decrementPrecedence(CurrentPrecedence);
    auto RhsPrecedence = CurrentPrecedence;

    // Assignment operators are right-associative.
    if (CurrentPrecedence == OperatorPrecedence::Assignment)
      std::swap(LhsPrecedence, RhsPrecedence);

    CurrentPrecedence = LhsPrecedence;
    rc_recur emitExpression(Op->getOperand(0));

    if (not mlir::isa<CommaOp>(Op))
      Out << ' ';

    Out << C.getOperator(getOperator(Op)) << ' ';

    CurrentPrecedence = RhsPrecedence;
    rc_recur emitExpression(Op->getOperand(1));
  }

  struct ExpressionEmitInfo {
    OperatorPrecedence Precedence;
    RecursiveCoroutine<void> (CEmitter::*Emit)(mlir::Value V);
  };

  // This function handles the dispatching for emitting different kinds of
  // expressions. It returns the precedence of the expression and a pointer to
  // a member function used for emitting it. The actual emission is only handled
  // afterwards. The reason for this is that the precedence must be known before
  // we start emitting the expression, because it may need to parenthesized.
  static ExpressionEmitInfo getExpressionEmitInfo(mlir::Value V) {
    auto E = V.getDefiningOp<ExpressionOpInterface>();

    if (not E) {
      if (mlir::isa<mlir::BlockArgument>(V)) {
        return {
          .Precedence = OperatorPrecedence::Primary,
          .Emit = &CEmitter::emitParameterExpression,
        };
      }

      if (auto Variable = V.getDefiningOp<LocalVariableOp>()) {
        return {
          .Precedence = OperatorPrecedence::Primary,
          .Emit = &CEmitter::emitLocalVariableExpression,
        };
      }

      revng_abort("This operation is not supported.");
    }

    if (mlir::isa<ImmediateOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Primary,
        .Emit = &CEmitter::emitImmediateExpression,
      };
    }

    if (mlir::isa<UseOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Primary,
        .Emit = &CEmitter::emitUseExpression,
      };
    }

    if (mlir::isa<AccessOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CEmitter::emitAccessExpression,
      };
    }

    if (mlir::isa<SubscriptOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CEmitter::emitSubscriptExpression,
      };
    }

    if (mlir::isa<CallOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CEmitter::emitCallExpression,
      };
    }

    if (mlir::isa<PostIncrementOp, PostDecrementOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CEmitter::emitPostfixExpression,
      };
    }

    if (auto Cast = mlir::dyn_cast<CastOp>(E.getOperation())) {
      if (Cast.getKind() == CastKind::Decay) {
        return {
          .Precedence = OperatorPrecedence::Primary,
          .Emit = &CEmitter::emitCastExpression,
        };
      }

      return {
        .Precedence = OperatorPrecedence::UnaryPrefix,
        .Emit = &CEmitter::emitCastExpression,
      };
    }

    if (mlir::isa<NegOp,
                  BitwiseNotOp,
                  LogicalNotOp,
                  IncrementOp,
                  DecrementOp,
                  AddressofOp,
                  IndirectionOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPrefix,
        .Emit = &CEmitter::emitPrefixExpression,
      };
    }

    if (mlir::isa<MulOp, DivOp, RemOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Multiplicative,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<AddOp, SubOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Additive,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<ShiftLeftOp, ShiftRightOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Shift,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<LessThanOp,
                  GreaterThanOp,
                  LessThanOrEqualOp,
                  GreaterThanOrEqualOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Relational,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<EqualOp, NotEqualOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Equality,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<BitwiseAndOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Bitand,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<BitwiseXorOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Bitxor,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<BitwiseOrOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Bitor,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<LogicalAndOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::And,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<LogicalOrOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Or,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<AssignOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Assignment,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<CommaOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Comma,
        .Emit = &CEmitter::emitInfixExpression,
      };
    }

    revng_abort("This operation is not supported.");
  }

  RecursiveCoroutine<void> emitExpression(mlir::Value V) {
    const ExpressionEmitInfo Info = getExpressionEmitInfo(V);

    bool PrintParentheses = Info.Precedence <= CurrentPrecedence
                            and Info.Precedence != OperatorPrecedence::Primary;

    if (PrintParentheses)
      Out << '(';

    // CurrentPrecedence is changed within this scope:
    {
      const auto PreviousPrecedence = CurrentPrecedence;
      const auto PrecedenceGuard = llvm::make_scope_exit([&]() {
        CurrentPrecedence = PreviousPrecedence;
      });
      CurrentPrecedence = Info.Precedence;

      // Emit the expression using the member function returned by
      // getExpressionEmitInfo.
      rc_recur(this->*Info.Emit)(V);
    }

    if (PrintParentheses)
      Out << ')';
  }

  RecursiveCoroutine<void> emitExpressionRegion(mlir::Region &R) {
    revng_assert(R.hasOneBlock());
    mlir::Block &B = R.front();

    auto End = B.end();
    revng_assert(End != B.begin());

    auto Yield = mlir::cast<YieldOp>(*--End);
    return emitExpression(Yield.getValue());
  }

  //===---------------------------- Statements ----------------------------===//

  RecursiveCoroutine<void> emitKeywordStatement(Keyword K) {
    Out << C.getKeyword(K) << ';' << '\n';
    rc_return;
  }

  RecursiveCoroutine<void> emitLocalVariableDeclaration(LocalVariableOp S) {
    // TODO: Emit variable name from the model once the model is extended to
    //       provide this information.

    auto Symbol = getLocalSymbolName(S);
    rc_recur emitDeclaration(S.getResult().getType(),
                             C.getVariableLocationDefinition(Symbol,
                                                             *CurrentFunction));

    if (not S.getInitializer().empty()) {
      Out << ' ' << '=' << ' ';

      // Comma expressions in a variable initialiser must be parenthesized.
      CurrentPrecedence = OperatorPrecedence::Comma;

      rc_recur emitExpressionRegion(S.getInitializer());
    }

    Out << ';' << '\n';
  }

  RecursiveCoroutine<void> emitLabelStatement(AssignLabelOp S) {
    Out.unindent();

    // TODO: Emit the label name from the model once the model is extended to
    //       provide this information.
    auto Symbol = getLocalSymbolName(S.getLabelOp());
    Out << C.getGotoLabelLocationDefinition(Symbol, *CurrentFunction) << ':';

    // Until C23, labels cannot be placed at the end of a block.
    if (S.getOperation() == &S->getBlock()->back())
      Out << ' ' << ';';

    Out << '\n';

    Out.indent();
    rc_return;
  }

  RecursiveCoroutine<void> emitExpressionStatement(ExpressionStatementOp S) {
    rc_recur emitExpressionRegion(S.getExpression());
    Out << ';' << '\n';
  }

  RecursiveCoroutine<void> emitGotoStatement(GoToOp S) {
    // TODO: Emit the label name from the model once the model is extended to
    //       provide this information.

    auto Symbol = getLocalSymbolName(S.getLabelOp());
    Out << C.getKeyword(Keyword::Goto) << ' '
        << C.getGotoLabelLocationReference(Symbol, *CurrentFunction) << ';'
        << '\n';

    rc_return;
  }

  RecursiveCoroutine<void> emitReturnStatement(ReturnOp S) {
    Out << C.getKeyword(Keyword::Return) << ' ';

    if (not S.getResult().empty())
      rc_recur emitExpressionRegion(S.getResult());

    Out << ';' << '\n';
  }

  static bool mayElideIfStatementBraces(IfOp If) {
    while (true) {
      if (not mayElideBraces(If.getThen()))
        return false;

      if (If.getElse().empty())
        return true;

      auto ElseIf = getOnlyOperation<IfOp>(If.getElse());

      if (not ElseIf)
        return mayElideBraces(If.getElse());

      If = ElseIf;
    }
  }

  RecursiveCoroutine<void> emitIfStatement(IfOp S) {
    // Nested if-else-if chains are printed out in a loop to avoid introducing
    // extra indentation for each else-if.

    bool EmitBlocks = not mayElideIfStatementBraces(S);

    while (true) {
      Out << C.getKeyword(Keyword::If) << ' ' << '(';
      rc_recur emitExpressionRegion(S.getCondition());
      Out << ')';

      rc_recur emitImplicitBlockStatement(S.getThen(), EmitBlocks);

      if (S.getElse().empty())
        break;

      if (EmitBlocks)
        Out << ' ';

      Out << C.getKeyword(Keyword::Else);

      if (auto ElseIf = getOnlyOperation<IfOp>(S.getElse())) {
        S = ElseIf;
        Out << ' ';
      } else {
        rc_recur emitImplicitBlockStatement(S.getElse(), EmitBlocks);

        if (EmitBlocks)
          Out << '\n';

        break;
      }
    }
  }

  RecursiveCoroutine<void> emitSwitchStatement(SwitchOp S) {
    Out << C.getKeyword(Keyword::Switch) << ' ' << '(';
    rc_recur emitExpressionRegion(S.getCondition());
    Out << ')' << ' ';

    // Scope tags are applied within this scope:
    {
      Scope Scope(Out);

      ValueType Type = S.getConditionType();
      for (unsigned I = 0, Count = S.getNumCases(); I < Count; ++I) {
        Out << C.getKeyword(Keyword::Case) << ' ';
        emitIntegerImmediate(S.getCaseValue(I), Type);
        Out << ':';
        if (rc_recur emitImplicitBlockStatement(S.getCaseRegion(I)))
          Out << '\n';
      }

      if (S.hasDefaultCase()) {
        Out << C.getKeyword(Keyword::Default) << ':';
        if (rc_recur emitImplicitBlockStatement(S.getDefaultCaseRegion()))
          Out << '\n';
      }
    }

    Out << '\n';
  }

  RecursiveCoroutine<void> emitForStatement(ForOp S) {
    Out << C.getKeyword(Keyword::For) << ' ' << '(' << ';';

    if (not S.getCondition().empty()) {
      Out << ' ';
      rc_recur emitExpressionRegion(S.getCondition());
    }

    Out << ';';
    if (not S.getExpression().empty()) {
      Out << ' ';
      rc_recur emitExpressionRegion(S.getExpression());
    }
    Out << ')';

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      Out << '\n';
  }

  RecursiveCoroutine<void> emitWhileStatement(WhileOp S) {
    Out << C.getKeyword(Keyword::While) << ' ' << '(';
    rc_recur emitExpressionRegion(S.getCondition());
    Out << ')';

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      Out << '\n';
  }

  RecursiveCoroutine<void> emitDoWhileStatement(DoWhileOp S) {
    Out << C.getKeyword(Keyword::Do);

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      Out << ' ';

    Out << C.getKeyword(Keyword::While) << ' ' << '(';
    rc_recur emitExpressionRegion(S.getCondition());
    Out << ')' << ';' << '\n';
  }

  RecursiveCoroutine<void> emitStatement(StatementOpInterface Stmt) {
    mlir::Operation *Op = Stmt.getOperation();

    if (auto S = mlir::dyn_cast<LocalVariableOp>(Op))
      return emitLocalVariableDeclaration(S);

    if (auto S = mlir::dyn_cast<MakeLabelOp>(Op))
      return noopCoroutine();

    if (auto S = mlir::dyn_cast<AssignLabelOp>(Op))
      return emitLabelStatement(S);

    if (auto S = mlir::dyn_cast<ExpressionStatementOp>(Op))
      return emitExpressionStatement(S);

    if (auto S = mlir::dyn_cast<GoToOp>(Op))
      return emitGotoStatement(S);

    if (mlir::isa<LoopBreakOp, SwitchBreakOp>(Op))
      return emitKeywordStatement(Keyword::Break);

    if (mlir::isa<LoopContinueOp>(Op))
      return emitKeywordStatement(Keyword::Continue);

    if (auto S = mlir::dyn_cast<ReturnOp>(Op))
      return emitReturnStatement(S);

    if (auto S = mlir::dyn_cast<IfOp>(Op))
      return emitIfStatement(S);

    if (auto S = mlir::dyn_cast<SwitchOp>(Op))
      return emitSwitchStatement(S);

    if (auto S = mlir::dyn_cast<ForOp>(Op))
      return emitForStatement(S);

    if (auto S = mlir::dyn_cast<WhileOp>(Op))
      return emitWhileStatement(S);

    if (auto S = mlir::dyn_cast<DoWhileOp>(Op))
      return emitDoWhileStatement(S);

    revng_abort("Unsupported operation");
  }

  RecursiveCoroutine<void> emitStatementRegion(mlir::Region &R) {
    for (mlir::Operation &Stmt : R.getOps())
      rc_recur emitStatement(mlir::cast<StatementOpInterface>(&Stmt));
  }

  static bool mayElideBraces(mlir::Operation *Op) {
    return mlir::isa<ExpressionStatementOp,
                     ReturnOp,
                     SwitchBreakOp,
                     LoopBreakOp,
                     LoopContinueOp>(Op);
  }

  static bool mayElideBraces(mlir::Region &R) {
    mlir::Operation *OnlyOp = getOnlyOperation(R);
    return OnlyOp != nullptr and mayElideBraces(OnlyOp);
  }

  RecursiveCoroutine<void> emitImplicitBlockStatement(mlir::Region &R,
                                                      bool EmitBlock) {
    std::optional<PairedScope<"{", "}">> BraceScope;

    if (EmitBlock) {
      Out << ' ';
      BraceScope.emplace(Out);
    }

    auto Scope = C.scopeTag(ptml::c::scopes::Scope).scope(Out);
    ptml::IndentedOstream::Scope IndentScope(Out);

    Out << '\n';
    rc_recur emitStatementRegion(R);
  }

  RecursiveCoroutine<bool> emitImplicitBlockStatement(mlir::Region &R) {
    bool EmitBlock = not mayElideBraces(R);
    rc_recur emitImplicitBlockStatement(R, EmitBlock);
    rc_return EmitBlock;
  }

  //===----------------------------- Functions ----------------------------===//

  RecursiveCoroutine<void> emitFunction(FunctionOp Op) {
    const model::Function &ModelFunction = getModelFunction(Op);
    CurrentFunction = &ModelFunction;

    auto *MFT = llvm::cast<model::DefinedType>(ModelFunction.Prototype().get());
    const model::TypeDefinition *MFD = MFT->Definition().get();

    auto ClearParameterNames = llvm::make_scope_exit([&]() {
      ParameterNames.clear();
    });

    auto PushParameterName = [&](llvm::StringRef ParameterName) {
      ParameterNames.push_back(C.getArgumentLocationReference(ParameterName,
                                                              ModelFunction));
    };

    if (auto F = llvm::dyn_cast<model::CABIFunctionDefinition>(MFD)) {
      for (const model::Argument &Parameter : F->Arguments())
        PushParameterName(C.NameBuilder.argumentName(*F, Parameter));
    } else if (auto F = llvm::dyn_cast<model::RawFunctionDefinition>(MFD)) {
      for (const model::NamedTypedRegister &Register : F->Arguments())
        PushParameterName(C.NameBuilder.argumentName(*F, Register));

      if (not F->StackArgumentsType().isEmpty())
        PushParameterName("_stack_arguments");
    } else {
      revng_abort("Unsupported model function type definition");
    }

    LocalVariableCounter = 0;
    GotoLabelCounter = 0;

    auto ClearLocalSymbols = llvm::make_scope_exit([&]() {
      LocalSymbolNames.clear();
    });

    // Scope tags are applied within this scope:
    {
      auto OuterScope = C.scopeTag(ptml::c::scopes::Function).scope(Out);
      C.printFunctionPrototype(*MFD, ModelFunction, /*SingleLine=*/false);

      Out << ' ';
      Scope InnerScope(Out, ptml::c::scopes::FunctionBody);

      if (const model::Type *T = ModelFunction.StackFrameType().get()) {
        const auto *D = llvm::cast<model::DefinedType>(T)->Definition().get();

        if (C.shouldInline(D->key()))
          C.printTypeDefinition(*D);
      }

      rc_recur emitStatementRegion(Op.getBody());
    }

    Out << '\n';
  }

private:
  const TargetCImplementation &Target;
  ptml::CTypeBuilder &C;
  ptml::IndentedOstream Out;

  const model::Function *CurrentFunction = nullptr;

  // Parameter names of the current function.
  llvm::SmallVector<std::string> ParameterNames;

  // Ambient precedence of the current expression.
  OperatorPrecedence CurrentPrecedence = {};

  size_t LocalVariableCounter = 0;
  size_t GotoLabelCounter = 0;

  llvm::DenseMap<mlir::Operation *, std::string> LocalSymbolNames;
};

} // namespace

std::string clift::decompile(FunctionOp Function,
                             const TargetCImplementation &Target,
                             ptml::CTypeBuilder &Builder) {
  std::string Result;
  llvm::raw_string_ostream Out(Result);
  CEmitter(Target, Builder, Out).emitFunction(Function);
  return Result;
}
