//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <variant>

#include "llvm/ADT/ScopeExit.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Pipeline/Location.h"
#include "revng/Pipes/Ranks.h"
#include "revng/mlir/Dialect/Clift/Utils/CBackend.h"

namespace clift = mlir::clift;

using namespace mlir::clift;

namespace {

static RecursiveCoroutine<void> noopCoroutine() {
  rc_return;
}

template<typename Operation = mlir::Operation *>
static Operation getOnlyOperation(mlir::Region &R) {
  if (R.empty())
    return {};

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

static bool hasFallthrough(mlir::Region &R) {
  // TODO: Refactor the logic of getting the last statement operation in a
  // region into a separate getTrailingStatement helper function.

  if (R.empty())
    return true;

  mlir::Block &B = R.front();
  if (B.empty())
    return true;

  return not B.back().hasTrait<mlir::OpTrait::clift::NoFallthrough>();
}

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

  Ternary = Assignment,
};

static std::string getPrimitiveTypeCName(PrimitiveKind Kind, uint64_t Size) {
  auto GetPrefix = [](PrimitiveKind Kind) -> llvm::StringRef {
    switch (Kind) {
    case PrimitiveKind::UnsignedKind:
      return "uint";
    case PrimitiveKind::SignedKind:
      return "int";
    default:
      return clift::stringifyPrimitiveKind(Kind);
    }
  };

  std::string Name;
  {
    llvm::raw_string_ostream Out(Name);
    Out << GetPrefix(Kind);

    if (Kind != PrimitiveKind::VoidKind)
      Out << (Size * 8) << "_t";
  }
  return Name;
}

class CliftToCEmitter {
  using CTE = CTokenEmitter;

  CTokenEmitter &Emitter;
  const TargetCImplementation &Target;

  FunctionOp CurrentFunction = {};
  pipeline::Location<decltype(revng::ranks::Function)> CurrentFunctionLocation;

  // Ambient precedence of the current expression.
  OperatorPrecedence CurrentPrecedence = {};

public:
  explicit CliftToCEmitter(CTokenEmitter &Emitter,
                           const TargetCImplementation &Target) :
    Emitter(Emitter), Target(Target) {}

  llvm::StringRef getStringAttr(mlir::Operation *Op, llvm::StringRef Name) {
    return mlir::cast<mlir::StringAttr>(Op->getAttr(Name)).getValue();
  }

  llvm::StringRef getNameAttr(mlir::Operation *Op) {
    return getStringAttr(Op, "clift.name");
  }

  llvm::StringRef getLocationAttr(mlir::Operation *Op) {
    return getStringAttr(Op, "clift.handle");
  }

  void emitPrimitiveType(PrimitiveKind Kind, uint64_t Size) {
    if (Kind == PrimitiveKind::VoidKind) {
      Emitter.emitKeyword(CTE::Keyword::Void);
    } else {
      auto TypeName = getPrimitiveTypeCName(Kind, Size);
      auto Location = pipeline::locationString(revng::ranks::PrimitiveType,
                                               TypeName);

      Emitter.emitIdentifier(TypeName,
                             Location,
                             CTE::EntityKind::Primitive,
                             CTE::IdentifierKind::Reference);
    }
  }

  void emitPrimitiveType(PrimitiveType Type) {
    emitPrimitiveType(Type.getKind(), Type.getSize());
  }

  /// Describes a function parameter declarator.
  struct ParameterDeclaratorInfo {
    llvm::StringRef Identifier;
    llvm::StringRef Location;
  };

  /// Describes a declarator. This can be any function or variable declarator,
  /// including a function parameter declarator. When emitting a function
  /// declaration, the parameters declarators array must contain entries for
  /// each parameter of the outermost function type.
  struct DeclaratorInfo {
    llvm::StringRef Identifier;
    llvm::StringRef Location;
    CTE::EntityKind Kind;

    llvm::ArrayRef<ParameterDeclaratorInfo> Parameters;
  };

  RecursiveCoroutine<void>
  emitDeclaration(ValueType Type, std::optional<DeclaratorInfo> Declarator) {
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
        Emitter.emitSpace();
      NeedSpace = false;
    };

    auto EmitConst = [&](ValueType T) {
      EmitSpace();
      if (T.isConst()) {
        Emitter.emitKeyword(CTE::Keyword::Const);
        Emitter.emitSpace();
      }
    };

    // Expanded function parameter declarator names are only emitted for the
    // outermost function type of a function declarator. When emitting a
    // function declarator, this optional is engaged. It is initially null, and
    // is filled out by the first function type visited.
    std::optional<FunctionType> OutermostFunctionType;
    if (Declarator and Declarator->Kind == CTE::EntityKind::Function)
      OutermostFunctionType.emplace(nullptr);

    auto IsOutermostFunction = [&](FunctionType F) {
      if (not OutermostFunctionType)
        return false;

      if (*OutermostFunctionType == nullptr)
        OutermostFunctionType.emplace(F);

      return F == *OutermostFunctionType;
    };

    // Recurse through the declaration, pushing each level onto the stack until
    // a terminal type is encountered. Primitive types as well as defined types
    // are considered terminal. Function types are not considered terminal if
    // function type expansion is enabled. Pointers with size not matching the
    // pointer size of the target implementation are considered terminal and
    // are printed by recursively entering this function.
    while (true) {
      StackItem Item = { StackItemKind::Terminal, Type };

      if (auto T = mlir::dyn_cast<PrimitiveType>(Type)) {
        EmitConst(T);
        emitPrimitiveType(T);
        NeedSpace = true;
      } else if (auto T = mlir::dyn_cast<PointerType>(Type)) {
        if (T.getPointerSize() == Target.PointerSize) {
          Item.Kind = StackItemKind::Pointer;
          Type = T.getPointeeType();
        } else {
          std::string Name;
          {
            llvm::raw_string_ostream Out(Name);
            Out << "pointer" << (T.getPointerSize() * 8) << "_t";
          }

          EmitConst(T);
          Emitter.emitLiteralIdentifier(Name);
          Emitter.emitPunctuator(CTE::Punctuator::LeftParenthesis);

          rc_recur emitType(T.getPointeeType());

          Emitter.emitPunctuator(CTE::Punctuator::RightParenthesis);
          NeedSpace = true;
        }
      } else if (auto T = mlir::dyn_cast<ArrayType>(Type)) {
        Item.Kind = StackItemKind::Array;
        Type = T.getElementType();
      } else if (auto T = mlir::dyn_cast<DefinedType>(Type)) {
        auto F = mlir::dyn_cast<FunctionType>(T);

        // Expand the function type if function type expansion is enabled.
        if (F and (ExpandFunctionTypes or IsOutermostFunction(F))) {
          Item.Kind = StackItemKind::Function;
          Type = F.getReturnType();
        } else {
          auto Kind = CTE::EntityKind::Typedef;

          if (mlir::isa<FunctionType>(T))
            Kind = CTE::EntityKind::Function;
          else if (mlir::isa<StructType>(T))
            Kind = CTE::EntityKind::Struct;
          else if (mlir::isa<UnionType>(T))
            Kind = CTE::EntityKind::Union;
          else if (mlir::isa<EnumType>(T))
            Kind = CTE::EntityKind::Enum;

          EmitConst(T);
          Emitter.emitIdentifier(T.getName(),
                                 T.getHandle(),
                                 Kind,
                                 CTE::IdentifierKind::Reference);

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
        EmitSpace();
        Emitter.emitPunctuator(CTE::Punctuator::Star);
        EmitConst(T);
      } break;
      case StackItemKind::Array: {
        if (I != 0 and Stack[I - 1].Kind != StackItemKind::Array) {
          Emitter.emitPunctuator(CTE::Punctuator::LeftParenthesis);
          NeedSpace = false;
        }
      } break;
      case StackItemKind::Function: {
        if (I != 0) {
          Emitter.emitPunctuator(CTE::Punctuator::LeftParenthesis);
          NeedSpace = false;
        }
      } break;
      }
    }

    if (Declarator) {
      EmitSpace();
      Emitter.emitIdentifier(Declarator->Identifier,
                             Declarator->Location,
                             Declarator->Kind,
                             CTE::IdentifierKind::Definition);
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
          Emitter.emitPunctuator(CTE::Punctuator::RightParenthesis);

        Emitter.emitPunctuator(CTE::Punctuator::LeftBracket);

        uint64_t Extent = mlir::cast<ArrayType>(SI.Type).getElementsCount();

        // Use a wider bit-width to handle the edge-case of an extent greater
        // than the maximum value of a signed 64-bit integer. Making the value
        // unsigned would cause unnecessary type suffixes to be emitted.
        auto ExtentValue = llvm::APSInt(llvm::APInt(/*numBits=*/128, Extent),
                                        /*isUnsigned=*/false);

        Emitter.emitIntegerLiteral(ExtentValue,
                                   CIntegerKind::Int,
                                   /*Radix=*/10);

        Emitter.emitPunctuator(CTE::Punctuator::RightBracket);
      } break;
      case StackItemKind::Function: {
        auto F = mlir::dyn_cast<FunctionType>(SI.Type);
        bool IsOutermost = IsOutermostFunction(F);

        if (I != 0)
          Emitter.emitPunctuator(CTE::Punctuator::RightParenthesis);

        Emitter.emitPunctuator(CTE::Punctuator::LeftParenthesis);
        if (F.getArgumentTypes().empty()) {
          emitPrimitiveType(PrimitiveKind::VoidKind, 0);
        } else {
          for (auto [J, PT] : llvm::enumerate(F.getArgumentTypes())) {
            if (J != 0) {
              Emitter.emitPunctuator(CTE::Punctuator::Comma);
              Emitter.emitSpace();
            }

            std::optional<DeclaratorInfo> ParameterDeclarator;
            if (IsOutermost) {
              ParameterDeclarator = DeclaratorInfo{
                .Identifier = Declarator->Parameters[J].Identifier,
                .Location = Declarator->Parameters[J].Location,
                .Kind = CTE::EntityKind::FunctionParameter,
              };
            }

            rc_recur emitDeclaration(PT, ParameterDeclarator);
          }
        }
        Emitter.emitPunctuator(CTE::Punctuator::RightParenthesis);
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

  static llvm::APSInt makeIntegerValue(PrimitiveType Type, uint64_t Value) {
    bool Signed = Type.getKind() == PrimitiveKind::SignedKind;

    return llvm::APSInt(llvm::APInt(Type.getSize() * 8, Value, Signed),
                        not Signed);
  }

  void emitIntegerImmediate(uint64_t Value, ValueType Type) {
    Type = dealias(Type, /*IgnoreQualifiers=*/true);

    if (auto T = mlir::dyn_cast<PrimitiveType>(Type)) {
      auto Integer = Target.getIntegerKind(T.getSize());

      if (not Integer) {
        // Emit explicit cast if the standard integer type is not known. Emit
        // the literal itself without a suffix (as if int).

        Emitter.emitOperator(CTE::Operator::LeftParenthesis);
        emitPrimitiveType(T);
        Emitter.emitOperator(CTE::Operator::RightParenthesis);

        Integer = CIntegerKind::Int;
      }

      Emitter.emitIntegerLiteral(makeIntegerValue(T, Value), *Integer, 10);
    } else {
      auto Enum = mlir::cast<EnumType>(Type);
      auto Enumerator = Enum.getFieldByValue(Value);

      Emitter.emitIdentifier(Enumerator.getName(),
                             Enumerator.getHandle(),
                             CTE::EntityKind::Enumerator,
                             CTE::IdentifierKind::Reference);
    }
  }

  //===---------------------------- Expressions ---------------------------===//

  RecursiveCoroutine<void> emitUndefExpression(mlir::Value V) {
    revng_assert(isScalarType(V.getType()));

    Emitter.emitLiteralIdentifier("undef");
    Emitter.emitOperator(CTE::Operator::LeftParenthesis);

    rc_recur emitType(V.getType());

    Emitter.emitOperator(CTE::Operator::RightParenthesis);
  }

  RecursiveCoroutine<void> emitImmediateExpression(mlir::Value V) {
    auto E = V.getDefiningOp<ImmediateOp>();
    emitIntegerImmediate(E.getValue(), E.getResult().getType());
    rc_return;
  }

  RecursiveCoroutine<void> emitStringLiteralExpression(mlir::Value V) {
    auto E = V.getDefiningOp<StringOp>();
    Emitter.emitStringLiteral(E.getValue());
    rc_return;
  }

  RecursiveCoroutine<void> emitAggregateInitializer(AggregateOp E) {
    // The precedence here must be comma, because an initializer list cannot
    // contain an unparenthesized comma expression. It would be parsed as two
    // initializers instead.
    CurrentPrecedence = OperatorPrecedence::Comma;

    Emitter.emitPunctuator(CTE::Punctuator::LeftBrace);
    for (auto [I, Initializer] : llvm::enumerate(E.getInitializers())) {
      if (I != 0) {
        Emitter.emitPunctuator(CTE::Punctuator::Comma);
        Emitter.emitSpace();
      }

      rc_recur emitExpression(Initializer);
    }
    Emitter.emitPunctuator(CTE::Punctuator::RightBrace);
  }

  RecursiveCoroutine<void> emitAggregateExpression(mlir::Value V) {
    auto E = V.getDefiningOp<AggregateOp>();

    Emitter.emitOperator(CTE::Operator::LeftParenthesis);
    rc_recur emitType(E.getResult().getType());
    Emitter.emitOperator(CTE::Operator::RightParenthesis);

    rc_recur emitAggregateInitializer(E);
  }

  RecursiveCoroutine<void> emitParameterExpression(mlir::Value V) {
    auto E = mlir::cast<mlir::BlockArgument>(V);

    const auto &ArgAttrs = CurrentFunction.getArgAttrs(E.getArgNumber());
    const auto GetStringAttr = [&ArgAttrs](llvm::StringRef Name) {
      return mlir::cast<mlir::StringAttr>(ArgAttrs.get(Name)).getValue();
    };

    Emitter.emitIdentifier(GetStringAttr("clift.name"),
                           GetStringAttr("clift.handle"),
                           CTE::EntityKind::FunctionParameter,
                           CTE::IdentifierKind::Reference);
    rc_return;
  }

  RecursiveCoroutine<void> emitLocalVariableExpression(mlir::Value V) {
    auto E = V.getDefiningOp<LocalVariableOp>();

    Emitter.emitIdentifier(getNameAttr(E),
                           E.getHandle(),
                           CTE::EntityKind::LocalVariable,
                           CTE::IdentifierKind::Reference);
    rc_return;
  }

  RecursiveCoroutine<void> emitUseExpression(mlir::Value V) {
    auto E = V.getDefiningOp<UseOp>();

    auto Module = E->getParentOfType<mlir::ModuleOp>();
    revng_assert(Module);

    auto S = mlir::SymbolTable::lookupSymbolIn(Module, E.getSymbolNameAttr());
    auto Symbol = mlir::cast<GlobalOpInterface>(S);

    constexpr auto GetEntityKind = [](GlobalOpInterface Symbol) {
      if (mlir::isa<FunctionOp>(Symbol))
        return CTE::EntityKind::Function;
      if (mlir::isa<GlobalVariableOp>(Symbol))
        return CTE::EntityKind::GlobalVariable;
      revng_abort("Unsupported global operation");
    };

    Emitter.emitIdentifier(Symbol.getName(),
                           Symbol.getHandle(),
                           GetEntityKind(Symbol),
                           CTE::IdentifierKind::Reference);

    rc_return;
  }

  RecursiveCoroutine<void> emitAccessExpression(mlir::Value V) {
    auto E = V.getDefiningOp<AccessOp>();

    // Parenthesizing a nested unary postfix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);

    rc_recur emitExpression(E.getValue());

    Emitter.emitOperator(E.isIndirect() ? CTE::Operator::Arrow :
                                          CTE::Operator::Dot);

    auto Field = E.getClassType().getFields()[E.getMemberIndex()];

    Emitter.emitIdentifier(Field.getName(),
                           Field.getHandle(),
                           CTE::EntityKind::Field,
                           CTE::IdentifierKind::Reference);
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

    Emitter.emitOperator(CTE::Operator::LeftBracket);
    rc_recur emitExpression(E.getIndex());
    Emitter.emitOperator(CTE::Operator::RightBracket);
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

    Emitter.emitOperator(CTE::Operator::LeftParenthesis);
    for (auto [I, A] : llvm::enumerate(E.getArguments())) {
      if (I != 0) {
        Emitter.emitPunctuator(CTE::Punctuator::Comma);
        Emitter.emitSpace();
      }

      rc_recur emitExpression(A);
    }
    Emitter.emitOperator(CTE::Operator::RightParenthesis);
  }

  static bool isHiddenCast(CastOp Cast) {
    return Cast.getKind() == CastKind::Decay;
  }

  static mlir::Value unwrapHiddenCasts(CastOp Cast) {
    revng_assert(isHiddenCast(Cast));

    while (true) {
      auto InnerCast = Cast.getValue().getDefiningOp<CastOp>();
      if (not InnerCast or not isHiddenCast(InnerCast))
        break;
    }

    return Cast.getValue();
  }

  RecursiveCoroutine<void> emitCastExpression(mlir::Value V) {
    auto E = V.getDefiningOp<CastOp>();

    Emitter.emitOperator(CTE::Operator::LeftParenthesis);
    rc_recur emitType(E.getResult().getType());
    Emitter.emitOperator(CTE::Operator::RightParenthesis);

    // Parenthesizing a nested unary prefix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPrefix);

    rc_recur emitExpression(E.getValue());
  }

  RecursiveCoroutine<void> emitHiddenCastExpression(mlir::Value V) {
    return emitExpression(unwrapHiddenCasts(V.getDefiningOp<CastOp>()));
  }

  RecursiveCoroutine<void> emitTernaryExpression(mlir::Value V) {
    auto E = V.getDefiningOp<TernaryOp>();

    rc_recur emitExpression(E.getCondition());

    Emitter.emitSpace();
    Emitter.emitOperator(CTE::Operator::Question);
    Emitter.emitSpace();

    rc_recur emitExpression(E.getLhs());

    Emitter.emitSpace();
    Emitter.emitOperator(CTE::Operator::Colon);
    Emitter.emitSpace();

    // The right hand expression does not need parentheses.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::Ternary);

    rc_recur emitExpression(E.getRhs());
  }

  static CTE::Operator getOperator(mlir::Operation *Op) {
    if (mlir::isa<NegOp, SubOp, PtrSubOp, PtrDiffOp>(Op))
      return CTE::Operator::Minus;
    if (mlir::isa<AddOp, PtrAddOp>(Op))
      return CTE::Operator::Plus;
    if (mlir::isa<MulOp, IndirectionOp>(Op))
      return CTE::Operator::Star;
    if (mlir::isa<DivOp>(Op))
      return CTE::Operator::Slash;
    if (mlir::isa<RemOp>(Op))
      return CTE::Operator::Percent;
    if (mlir::isa<LogicalNotOp>(Op))
      return CTE::Operator::Exclaim;
    if (mlir::isa<LogicalAndOp>(Op))
      return CTE::Operator::AmpersandAmpersand;
    if (mlir::isa<LogicalOrOp>(Op))
      return CTE::Operator::PipePipe;
    if (mlir::isa<BitwiseNotOp>(Op))
      return CTE::Operator::Tilde;
    if (mlir::isa<BitwiseAndOp, AddressofOp>(Op))
      return CTE::Operator::Ampersand;
    if (mlir::isa<BitwiseOrOp>(Op))
      return CTE::Operator::Pipe;
    if (mlir::isa<BitwiseXorOp>(Op))
      return CTE::Operator::Caret;
    if (mlir::isa<ShiftLeftOp>(Op))
      return CTE::Operator::LessLess;
    if (mlir::isa<ShiftRightOp>(Op))
      return CTE::Operator::GreaterGreater;
    if (mlir::isa<CmpEqOp>(Op))
      return CTE::Operator::EqualsEquals;
    if (mlir::isa<CmpNeOp>(Op))
      return CTE::Operator::ExclaimEquals;
    if (mlir::isa<CmpLtOp>(Op))
      return CTE::Operator::Less;
    if (mlir::isa<CmpGtOp>(Op))
      return CTE::Operator::Greater;
    if (mlir::isa<CmpLeOp>(Op))
      return CTE::Operator::LessEquals;
    if (mlir::isa<CmpGeOp>(Op))
      return CTE::Operator::GreaterEquals;
    if (mlir::isa<IncrementOp, PostIncrementOp>(Op))
      return CTE::Operator::PlusPlus;
    if (mlir::isa<DecrementOp, PostDecrementOp>(Op))
      return CTE::Operator::MinusMinus;
    if (mlir::isa<AssignOp>(Op))
      return CTE::Operator::Equals;
    if (mlir::isa<CommaOp>(Op))
      return CTE::Operator::Comma;
    revng_abort("This operation does not represent a C operator.");
  }

  RecursiveCoroutine<void> emitPrefixExpression(mlir::Value V) {
    mlir::Operation *Op = V.getDefiningOp();
    mlir::Value Operand = Op->getOperand(0);

    Emitter.emitOperator(getOperator(Op));

    auto StartsWithMinus = [](mlir::Value V) {
      if (mlir::isa<NegOp, DecrementOp>(V.getDefiningOp()))
        return true;

      if (auto I = V.getDefiningOp<ImmediateOp>()) {
        if (auto T = mlir::dyn_cast<PrimitiveType>(I.getResult().getType())) {
          if (T.getKind() == PrimitiveKind::SignedKind)
            return static_cast<int64_t>(I.getValue()) < 0;
        }
      }

      return false;
    };

    // Double negation requires a space in between to avoid being confused as
    // decrement. (- -x) vs (--x)
    //
    // Negation after a decrement requires a space in between to avoid being
    // confused as decrement after negation. (- --x) vs (---x)
    if (V.getDefiningOp<NegOp>() and StartsWithMinus(Operand))
      Emitter.emitSpace();

    // Parenthesizing a nested unary prefix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPrefix);

    return emitExpression(Operand);
  }

  RecursiveCoroutine<void> emitPostfixExpression(mlir::Value V) {
    mlir::Operation *Op = V.getDefiningOp();
    rc_recur emitExpression(Op->getOperand(0));

    // Parenthesizing a nested unary postfix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);

    Emitter.emitOperator(getOperator(Op));
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
      Emitter.emitSpace();

    Emitter.emitOperator(getOperator(Op));
    Emitter.emitSpace();

    CurrentPrecedence = RhsPrecedence;
    rc_recur emitExpression(Op->getOperand(1));
  }

  struct ExpressionEmitInfo {
    OperatorPrecedence Precedence;
    RecursiveCoroutine<void> (CliftToCEmitter::*Emit)(mlir::Value V);
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
          .Emit = &CliftToCEmitter::emitParameterExpression,
        };
      }

      if (auto Variable = V.getDefiningOp<LocalVariableOp>()) {
        return {
          .Precedence = OperatorPrecedence::Primary,
          .Emit = &CliftToCEmitter::emitLocalVariableExpression,
        };
      }

      revng_abort("This operation is not supported.");
    }

    if (mlir::isa<UndefOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Primary,
        .Emit = &CliftToCEmitter::emitUndefExpression,
      };
    }

    if (mlir::isa<ImmediateOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Primary,
        .Emit = &CliftToCEmitter::emitImmediateExpression,
      };
    }

    if (mlir::isa<StringOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Primary,
        .Emit = &CliftToCEmitter::emitStringLiteralExpression,
      };
    }

    if (mlir::isa<AggregateOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Primary,
        .Emit = &CliftToCEmitter::emitAggregateExpression,
      };
    }

    if (mlir::isa<UseOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Primary,
        .Emit = &CliftToCEmitter::emitUseExpression,
      };
    }

    if (mlir::isa<AccessOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CliftToCEmitter::emitAccessExpression,
      };
    }

    if (mlir::isa<SubscriptOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CliftToCEmitter::emitSubscriptExpression,
      };
    }

    if (mlir::isa<CallOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CliftToCEmitter::emitCallExpression,
      };
    }

    if (mlir::isa<PostIncrementOp, PostDecrementOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::UnaryPostfix,
        .Emit = &CliftToCEmitter::emitPostfixExpression,
      };
    }

    if (auto Cast = mlir::dyn_cast<CastOp>(E.getOperation())) {
      if (isHiddenCast(Cast)) {
        auto Info = getExpressionEmitInfo(unwrapHiddenCasts(Cast));

        return {
          .Precedence = decrementPrecedence(Info.Precedence),
          .Emit = &CliftToCEmitter::emitHiddenCastExpression,
        };
      }

      return {
        .Precedence = OperatorPrecedence::UnaryPrefix,
        .Emit = &CliftToCEmitter::emitCastExpression,
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
        .Emit = &CliftToCEmitter::emitPrefixExpression,
      };
    }

    if (mlir::isa<MulOp, DivOp, RemOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Multiplicative,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<AddOp, SubOp, PtrAddOp, PtrSubOp, PtrDiffOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Additive,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<ShiftLeftOp, ShiftRightOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Shift,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<CmpLtOp, CmpGtOp, CmpLeOp, CmpGeOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Relational,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<CmpEqOp, CmpNeOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Equality,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<BitwiseAndOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Bitand,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<BitwiseXorOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Bitxor,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<BitwiseOrOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Bitor,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<LogicalAndOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::And,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<LogicalOrOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Or,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<AssignOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Assignment,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<CommaOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Comma,
        .Emit = &CliftToCEmitter::emitInfixExpression,
      };
    }

    if (mlir::isa<TernaryOp>(E)) {
      return {
        .Precedence = OperatorPrecedence::Ternary,
        .Emit = &CliftToCEmitter::emitTernaryExpression,
      };
    }

    revng_abort("This operation is not supported.");
  }

  RecursiveCoroutine<void> emitExpression(mlir::Value V) {
    const ExpressionEmitInfo Info = getExpressionEmitInfo(V);

    bool PrintParentheses = Info.Precedence <= CurrentPrecedence
                            and Info.Precedence != OperatorPrecedence::Primary;

    if (PrintParentheses)
      Emitter.emitPunctuator(CTE::Punctuator::LeftParenthesis);

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
      Emitter.emitPunctuator(CTE::Punctuator::RightParenthesis);
  }

  RecursiveCoroutine<void> emitExpressionRegion(mlir::Region &R) {
    mlir::Value Value = getExpressionValue(R);
    revng_assert(Value);
    return emitExpression(Value);
  }

  //===---------------------------- Statements ----------------------------===//

  RecursiveCoroutine<void> emitLocalVariableDeclaration(LocalVariableOp S) {
    rc_recur emitDeclaration(S.getResult().getType(),
                             DeclaratorInfo{
                               .Identifier = getNameAttr(S),
                               .Location = S.getHandle(),
                               .Kind = CTE::EntityKind::LocalVariable,
                             });

    if (not S.getInitializer().empty()) {
      Emitter.emitSpace();
      Emitter.emitOperator(CTE::Operator::Equals);
      Emitter.emitSpace();

      // Comma expressions in a variable initialiser must be parenthesized.
      CurrentPrecedence = OperatorPrecedence::Comma;

      mlir::Value Expression = getExpressionValue(S.getInitializer());

      if (auto Aggregate = Expression.getDefiningOp<AggregateOp>())
        rc_recur emitAggregateInitializer(Aggregate);
      else
        rc_recur emitExpression(Expression);
    }

    Emitter.emitPunctuator(CTE::Punctuator::Semicolon);
    Emitter.emitNewline();
  }

  bool labelRequiresEmptyExpression(AssignLabelOp Op) {
    // Prior to C23, labels cannot be placed at the end of a block:
    if (Op.getOperation() == &Op->getBlock()->back())
      return true;

    // Prior to C23, labels cannot be placed preceding a declaration:
    if (mlir::isa<LocalVariableOp>(&*std::next(Op->getIterator())))
      return true;

    return false;
  }

  RecursiveCoroutine<void> emitLabelStatement(AssignLabelOp S) {
    auto Scope = Emitter.enterScope(CTE::ScopeKind::None,
                                    CTE::Delimiter::None,
                                    /*Indent=*/-1);

    Emitter.emitIdentifier(getNameAttr(S.getLabelOp()),
                           getLocationAttr(S.getLabelOp()),
                           CTE::EntityKind::Label,
                           CTE::IdentifierKind::Definition);

    Emitter.emitPunctuator(CTE::Punctuator::Colon);

    if (labelRequiresEmptyExpression(S)) {
      Emitter.emitSpace();
      Emitter.emitPunctuator(CTE::Punctuator::Semicolon);
    }

    Emitter.emitNewline();

    rc_return;
  }

  RecursiveCoroutine<void> emitExpressionStatement(ExpressionStatementOp S) {
    rc_recur emitExpressionRegion(S.getExpression());
    Emitter.emitPunctuator(CTE::Punctuator::Semicolon);
    Emitter.emitNewline();
  }

  RecursiveCoroutine<void> emitGotoStatement(GoToOp S) {
    Emitter.emitKeyword(CTE::Keyword::Goto);
    Emitter.emitSpace();

    Emitter.emitIdentifier(getNameAttr(S.getLabelOp()),
                           getLocationAttr(S.getLabelOp()),
                           CTE::EntityKind::Label,
                           CTE::IdentifierKind::Reference);

    Emitter.emitPunctuator(CTE::Punctuator::Semicolon);
    Emitter.emitNewline();

    rc_return;
  }

  RecursiveCoroutine<void> emitReturnStatement(ReturnOp S) {
    Emitter.emitKeyword(CTE::Keyword::Return);

    if (not S.getResult().empty()) {
      Emitter.emitSpace();
      rc_recur emitExpressionRegion(S.getResult());
    }

    Emitter.emitPunctuator(CTE::Punctuator::Semicolon);
    Emitter.emitNewline();
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
      Emitter.emitKeyword(CTE::Keyword::If);
      Emitter.emitSpace();
      Emitter.emitPunctuator(CTE::Punctuator::LeftParenthesis);
      rc_recur emitExpressionRegion(S.getCondition());
      Emitter.emitPunctuator(CTE::Punctuator::RightParenthesis);

      rc_recur emitImplicitBlockStatement(S.getThen(), EmitBlocks);

      if (S.getElse().empty())
        break;

      if (EmitBlocks)
        Emitter.emitSpace();

      Emitter.emitKeyword(CTE::Keyword::Else);

      if (auto ElseIf = getOnlyOperation<IfOp>(S.getElse())) {
        S = ElseIf;
        Emitter.emitSpace();
      } else {
        rc_recur emitImplicitBlockStatement(S.getElse(), EmitBlocks);
        break;
      }
    }

    if (EmitBlocks)
      Emitter.emitNewline();
  }

  RecursiveCoroutine<void> emitCaseRegion(mlir::Region &R) {
    bool Break = hasFallthrough(R);

    if (rc_recur emitImplicitBlockStatement(R)) {
      if (Break)
        Emitter.emitSpace();
      else
        Emitter.emitNewline();
    }

    if (Break) {
      Emitter.emitKeyword(CTE::Keyword::Break);
      Emitter.emitPunctuator(CTE::Punctuator::Semicolon);
      Emitter.emitNewline();
    }
  }

  RecursiveCoroutine<void> emitSwitchStatement(SwitchOp S) {
    Emitter.emitKeyword(CTE::Keyword::Switch);
    Emitter.emitSpace();
    Emitter.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());

    Emitter.emitPunctuator(CTE::Punctuator::RightParenthesis);
    Emitter.emitSpace();

    // Scope tags are applied within this scope:
    {
      auto Scope = Emitter.enterScope(CTE::ScopeKind::BlockStatement,
                                      CTE::Delimiter::Braces,
                                      /*Indented=*/false);

      ValueType Type = S.getConditionType();
      for (unsigned I = 0, Count = S.getNumCases(); I < Count; ++I) {
        Emitter.emitKeyword(CTE::Keyword::Case);
        Emitter.emitSpace();
        emitIntegerImmediate(S.getCaseValue(I), Type);
        Emitter.emitPunctuator(CTE::Punctuator::Colon);
        rc_recur emitCaseRegion(S.getCaseRegion(I));
      }

      if (S.hasDefaultCase()) {
        Emitter.emitKeyword(CTE::Keyword::Default);
        Emitter.emitPunctuator(CTE::Punctuator::Colon);
        rc_recur emitCaseRegion(S.getDefaultCaseRegion());
      }
    }

    Emitter.emitNewline();
  }

  RecursiveCoroutine<void> emitForStatement(ForOp S) {
    Emitter.emitKeyword(CTE::Keyword::For);
    Emitter.emitSpace();
    Emitter.emitPunctuator(CTE::Punctuator::LeftParenthesis);
    Emitter.emitPunctuator(CTE::Punctuator::Semicolon);

    if (not S.getCondition().empty()) {
      Emitter.emitSpace();
      rc_recur emitExpressionRegion(S.getCondition());
    }

    Emitter.emitPunctuator(CTE::Punctuator::Semicolon);
    if (not S.getExpression().empty()) {
      Emitter.emitSpace();
      rc_recur emitExpressionRegion(S.getExpression());
    }
    Emitter.emitPunctuator(CTE::Punctuator::RightParenthesis);

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      Emitter.emitNewline();
  }

  RecursiveCoroutine<void> emitWhileStatement(WhileOp S) {
    Emitter.emitKeyword(CTE::Keyword::While);
    Emitter.emitSpace();
    Emitter.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());
    Emitter.emitPunctuator(CTE::Punctuator::RightParenthesis);

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      Emitter.emitNewline();
  }

  RecursiveCoroutine<void> emitDoWhileStatement(DoWhileOp S) {
    Emitter.emitKeyword(CTE::Keyword::Do);

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      Emitter.emitSpace();

    Emitter.emitKeyword(CTE::Keyword::While);
    Emitter.emitSpace();
    Emitter.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());

    Emitter.emitPunctuator(CTE::Punctuator::RightParenthesis);
    Emitter.emitPunctuator(CTE::Punctuator::Semicolon);
    Emitter.emitNewline();
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
    return mlir::isa<ExpressionStatementOp, GoToOp, ReturnOp>(Op);
  }

  static bool mayElideBraces(mlir::Region &R) {
    mlir::Operation *OnlyOp = getOnlyOperation(R);
    return OnlyOp != nullptr and mayElideBraces(OnlyOp);
  }

  RecursiveCoroutine<void> emitImplicitBlockStatement(mlir::Region &R,
                                                      bool EmitBlock) {
    auto ScopeKind = CTE::ScopeKind::None;
    auto Delimiter = CTE::Delimiter::None;

    if (EmitBlock) {
      Emitter.emitSpace();
      ScopeKind = CTE::ScopeKind::BlockStatement;
      Delimiter = CTE::Delimiter::Braces;
    }

    auto Scope = Emitter.enterScope(ScopeKind, Delimiter);
    Emitter.emitNewline();

    rc_recur emitStatementRegion(R);
  }

  RecursiveCoroutine<bool> emitImplicitBlockStatement(mlir::Region &R) {
    bool EmitBlock = not mayElideBraces(R);
    rc_recur emitImplicitBlockStatement(R, EmitBlock);
    rc_return EmitBlock;
  }

  //===----------------------------- Functions ----------------------------===//

  RecursiveCoroutine<void> emitFunction(FunctionOp Op) {
    CurrentFunction = Op;

    // Scope tags are applied within this scope:
    {
      auto OuterScope = Emitter.enterScope(CTE::ScopeKind::FunctionDeclaration,
                                           CTE::Delimiter::None,
                                           /*Indented=*/false);

      llvm::SmallVector<ParameterDeclaratorInfo> ParameterDeclarators;
      for (unsigned I = 0; I < Op.getArgCount(); ++I) {
        auto Attrs = Op.getArgAttrs(I);

        auto GetStringAttr = [&](llvm::StringRef Name) {
          return mlir::cast<mlir::StringAttr>(Attrs.get(Name)).getValue();
        };

        ParameterDeclarators.emplace_back(GetStringAttr("clift.name"),
                                          GetStringAttr("clift.handle"));
      }

      rc_recur emitDeclaration(Op.getCliftFunctionType(),
                               DeclaratorInfo{
                                 .Identifier = Op.getName(),
                                 .Location = Op.getHandle(),
                                 .Kind = CTE::EntityKind::Function,
                                 .Parameters = ParameterDeclarators,
                               });

      Emitter.emitSpace();

      auto InnerScope = Emitter.enterScope(CTE::ScopeKind::FunctionDefinition,
                                           CTE::Delimiter::Braces);

      Emitter.emitNewline();

      // TODO: Re-enable stack frame inlining.

      rc_recur emitStatementRegion(Op.getBody());

      // TODO: emit a comment containing homeless variable names.
      //       See how old backend does it for reference.
    }

    Emitter.emitNewline();
  }
};

} // namespace

void clift::decompile(FunctionOp Function,
                      CTokenEmitter &Emitter,
                      const TargetCImplementation &Target) {
  CliftToCEmitter(Emitter, Target).emitFunction(Function);
}
