//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <variant>

#include "llvm/ADT/ScopeExit.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/PTML/CBuilder.h"
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

static std::string getPrimitiveTypeCName(PrimitiveType Type) {
  return getPrimitiveTypeCName(Type.getKind(), Type.getSize());
}

// TODO: No action lists or context locations are currently emitted.
//       Should this logic be moved to the UI instead?
class CliftToCEmitter {
public:
  explicit CliftToCEmitter(CEmitter &Emitter,
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
      Emitter.emitKeyword(CEmitter::Keyword::Void);
    } else {
      auto TypeName = getPrimitiveTypeCName(Kind, Size);
      auto Location = pipeline::locationString(revng::ranks::PrimitiveType,
                                               TypeName);

      Emitter.emitIdentifier(TypeName,
                             Location,
                             CEmitter::EntityKind::Primitive,
                             CEmitter::IdentifierKind::Reference);
    }
  }

  void emitPrimitiveType(PrimitiveType Type) {
    emitPrimitiveType(Type.getKind(), Type.getSize());
  }

  void emitAttribute(AttributeAttr Attribute) {
    auto Macro = Attribute.getMacro();

    Emitter.emitSpace();
    Emitter.emitIdentifier(Macro.getString(),
                           Macro.getHandle(),
                           CEmitter::EntityKind::Attribute,
                           CEmitter::IdentifierKind::Reference);

    if (auto Arguments = Attribute.getArguments()) {
      Emitter.emitPunctuator(CEmitter::Symbol::LeftParenthesis);

      for (auto [I, A] : llvm::enumerate(*Arguments)) {
        if (I != 0) {
          Emitter.emitPunctuator(CEmitter::Symbol::Comma);
          Emitter.emitSpace();
        }

        Emitter.emitIdentifier(A.getString(),
                               A.getHandle(),
                               CEmitter::EntityKind::AttributeArgument,
                               CEmitter::IdentifierKind::Reference);
      }

      Emitter.emitPunctuator(CEmitter::Symbol::RightParenthesis);
    }
  }

  static bool checkAttributeArray(mlir::ArrayAttr ArrayAttr) {
    auto IsAttributeAttr = [](mlir::Attribute Attr) {
      return mlir::isa<AttributeAttr>(Attr);
    };
    return std::ranges::all_of(ArrayAttr, IsAttributeAttr);
  }

  mlir::ArrayAttr getDeclarationOpAttributes(mlir::Operation *Op) {
    if (auto Attr = Op->getAttr("clift.attributes")) {
      auto ArrayAttr = mlir::cast<mlir::ArrayAttr>(Attr);
      revng_assert(checkAttributeArray(ArrayAttr));
      return ArrayAttr;
    }
    return {};
  }

  void emitAttributes(mlir::ArrayAttr Attributes) {
    if (Attributes) {
      for (mlir::Attribute Attr : Attributes)
        emitAttribute(mlir::cast<AttributeAttr>(Attr));
    }
  }

  struct BasicDeclaratorInfo {
    llvm::StringRef Identifier;
    llvm::StringRef Location;
    mlir::ArrayAttr Attributes;
  };

  struct DeclaratorInfo {
    llvm::StringRef Identifier;
    llvm::StringRef Location;
    mlir::ArrayAttr Attributes;
    CEmitter::EntityKind Kind;

    llvm::ArrayRef<BasicDeclaratorInfo> Parameters;
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
        Emitter.emitKeyword(CEmitter::Keyword::Const);
        Emitter.emitSpace();
      }
    };

    // Expanded function parameter declarator names are only emitted for the
    // outermost function type of a function declarator. When emitting a
    // function declarator, this optional is engaged. It is initially null, and
    // is filled out by the first function type visited.
    std::optional<FunctionType> OutermostFunctionType;
    if (Declarator and Declarator->Kind == CEmitter::EntityKind::Function)
      OutermostFunctionType.emplace(nullptr);

    auto IsOutermostFunction = [&](FunctionType F) {
      if (OutermostFunctionType) {
        if (not *OutermostFunctionType)
          OutermostFunctionType.emplace(F);

        return F == *OutermostFunctionType;
      }
      return false;
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
          Emitter.emitPunctuator(CEmitter::Symbol::LeftParenthesis);

          rc_recur emitType(T.getPointeeType());

          Emitter.emitPunctuator(CEmitter::Symbol::RightParenthesis);
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
          auto Kind = CEmitter::EntityKind::Typedef;

          EmitConst(T);
          Emitter.emitIdentifier(T.getName(),
                                 T.getHandle(),
                                 Kind,
                                 CEmitter::IdentifierKind::Reference);

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
        Emitter.emitPunctuator(CEmitter::Symbol::Star);
        EmitConst(T);
      } break;
      case StackItemKind::Array: {
        if (I != 0 and Stack[I - 1].Kind != StackItemKind::Array) {
          Emitter.emitPunctuator(CEmitter::Symbol::LeftParenthesis);
          NeedSpace = false;
        }
      } break;
      case StackItemKind::Function: {
        if (I != 0) {
          Emitter.emitPunctuator(CEmitter::Symbol::LeftParenthesis);
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
                             CEmitter::IdentifierKind::Definition);
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
          Emitter.emitPunctuator(CEmitter::Symbol::RightParenthesis);

        Emitter.emitPunctuator(CEmitter::Symbol::LeftBracket);

        uint64_t Extent = mlir::cast<ArrayType>(SI.Type).getElementsCount();

        // Use a wider bit-width to handle the edge-case of an extent greater
        // than the maximum value of a signed 64-bit integer. Making the value
        // unsigned would cause unnecessary type suffixes to be emitted.
        auto ExtentValue = llvm::APSInt(llvm::APInt(/*numBits=*/128, Extent),
                                        /*isUnsigned=*/false);

        Emitter.emitIntegerLiteral(ExtentValue,
                                   CIntegerKind::Int,
                                   /*Radix=*/10);

        Emitter.emitPunctuator(CEmitter::Symbol::RightBracket);
      } break;
      case StackItemKind::Function: {
        auto F = mlir::dyn_cast<FunctionType>(SI.Type);
        bool IsOutermost = IsOutermostFunction(F);

        if (I != 0)
          Emitter.emitPunctuator(CEmitter::Symbol::RightParenthesis);

        Emitter.emitPunctuator(CEmitter::Symbol::LeftParenthesis);
        if (F.getArgumentTypes().empty()) {
          emitPrimitiveType(PrimitiveKind::VoidKind, 0);
        } else {
          for (auto [J, PT] : llvm::enumerate(F.getArgumentTypes())) {
            if (J != 0) {
              Emitter.emitPunctuator(CEmitter::Symbol::Comma);
              Emitter.emitSpace();
            }

            std::optional<DeclaratorInfo> ParameterDeclarator;
            if (IsOutermost) {
              ParameterDeclarator = DeclaratorInfo{
                .Identifier = Declarator->Parameters[J].Identifier,
                .Location = Declarator->Parameters[J].Location,
                .Attributes = Declarator->Parameters[J].Attributes,
                .Kind = CEmitter::EntityKind::FunctionParameter,
              };
            }

            rc_recur emitDeclaration(PT, ParameterDeclarator);
          }
        }
        Emitter.emitPunctuator(CEmitter::Symbol::RightParenthesis);
      } break;
      }
    }

    if (Declarator)
      emitAttributes(Declarator->Attributes);
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

        Emitter.emitOperator(CEmitter::Symbol::LeftParenthesis);
        emitPrimitiveType(T);
        Emitter.emitOperator(CEmitter::Symbol::RightParenthesis);

        Integer = CIntegerKind::Int;
      }

      Emitter.emitIntegerLiteral(makeIntegerValue(T, Value), *Integer, 10);
    } else {
      auto Enum = mlir::cast<EnumType>(Type);
      auto Enumerator = Enum.getFieldByValue(Value);

      Emitter.emitIdentifier(Enumerator.getName(),
                             Enumerator.getHandle(),
                             CEmitter::EntityKind::Enumerator,
                             CEmitter::IdentifierKind::Reference);
    }
  }

  //===---------------------------- Expressions ---------------------------===//

  RecursiveCoroutine<void> emitUndefExpression(mlir::Value V) {
    auto T = mlir::cast<clift::PrimitiveType>(V.getType());

    std::string Name;
    {
      llvm::raw_string_ostream Out(Name);
      Out << Target.UndefFunctionPrefix << getPrimitiveTypeCName(T);
    }

    Emitter.emitLiteralIdentifier(Name);
    Emitter.emitOperator(CEmitter::Symbol::LeftParenthesis);
    Emitter.emitOperator(CEmitter::Symbol::RightParenthesis);

    rc_return;
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

    Emitter.emitPunctuator(CEmitter::Symbol::LeftBrace);
    for (auto [I, Initializer] : llvm::enumerate(E.getInitializers())) {
      if (I != 0) {
        Emitter.emitPunctuator(CEmitter::Symbol::Comma);
        Emitter.emitSpace();
      }

      rc_recur emitExpression(Initializer);
    }
    Emitter.emitPunctuator(CEmitter::Symbol::RightBrace);
  }

  RecursiveCoroutine<void> emitAggregateExpression(mlir::Value V) {
    auto E = V.getDefiningOp<AggregateOp>();

    Emitter.emitOperator(CEmitter::Symbol::LeftParenthesis);
    rc_recur emitType(E.getResult().getType());
    Emitter.emitOperator(CEmitter::Symbol::RightParenthesis);

    rc_recur emitAggregateInitializer(E);
  }

  RecursiveCoroutine<void> emitParameterExpression(mlir::Value V) {
    auto E = mlir::cast<mlir::BlockArgument>(V);

    const auto &ArgAttrs = CurrentFunction.getArgAttrs(E.getArgNumber());
    const auto GetStringAttr = [&](llvm::StringRef Name) {
      return mlir::cast<mlir::StringAttr>(ArgAttrs.get(Name)).getValue();
    };

    Emitter.emitIdentifier(GetStringAttr("clift.name"),
                           GetStringAttr("clift.handle"),
                           CEmitter::EntityKind::FunctionParameter,
                           CEmitter::IdentifierKind::Reference);
    rc_return;
  }

  RecursiveCoroutine<void> emitLocalVariableExpression(mlir::Value V) {
    auto E = V.getDefiningOp<LocalVariableOp>();

    Emitter.emitIdentifier(getNameAttr(E),
                           E.getHandle(),
                           CEmitter::EntityKind::LocalVariable,
                           CEmitter::IdentifierKind::Reference);
    rc_return;
  }

  RecursiveCoroutine<void> emitUseExpression(mlir::Value V) {
    auto E = V.getDefiningOp<UseOp>();

    auto Module = E->getParentOfType<mlir::ModuleOp>();
    revng_assert(Module);

    auto S = mlir::SymbolTable::lookupSymbolIn(Module, E.getSymbolNameAttr());
    auto Symbol = mlir::cast<GlobalOpInterface>(S);

    auto GetEntityKind = [&]() {
      if (mlir::isa<FunctionOp>(Symbol))
        return CEmitter::EntityKind::Function;
      if (mlir::isa<GlobalVariableOp>(Symbol))
        return CEmitter::EntityKind::GlobalVariable;
      revng_abort("Unsupported global operation");
    };

    Emitter.emitIdentifier(Symbol.getName(),
                           Symbol.getHandle(),
                           GetEntityKind(),
                           CEmitter::IdentifierKind::Reference);

    rc_return;
  }

  RecursiveCoroutine<void> emitAccessExpression(mlir::Value V) {
    auto E = V.getDefiningOp<AccessOp>();

    // Parenthesizing a nested unary postfix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);

    rc_recur emitExpression(E.getValue());

    Emitter.emitOperator(E.isIndirect() ? CEmitter::Symbol::Arrow :
                                          CEmitter::Symbol::Dot);

    auto Field = E.getClassType().getFields()[E.getMemberIndex()];

    Emitter.emitIdentifier(Field.getName(),
                           Field.getHandle(),
                           CEmitter::EntityKind::Field,
                           CEmitter::IdentifierKind::Reference);
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

    Emitter.emitOperator(CEmitter::Symbol::LeftBracket);
    rc_recur emitExpression(E.getIndex());
    Emitter.emitOperator(CEmitter::Symbol::RightBracket);
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

    Emitter.emitOperator(CEmitter::Symbol::LeftParenthesis);
    for (auto [I, A] : llvm::enumerate(E.getArguments())) {
      if (I != 0) {
        Emitter.emitPunctuator(CEmitter::Symbol::Comma);
        Emitter.emitSpace();
      }

      rc_recur emitExpression(A);
    }
    Emitter.emitOperator(CEmitter::Symbol::RightParenthesis);
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

    Emitter.emitOperator(CEmitter::Symbol::LeftParenthesis);
    rc_recur emitType(E.getResult().getType());
    Emitter.emitOperator(CEmitter::Symbol::RightParenthesis);

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
    Emitter.emitOperator(CEmitter::Symbol::Question);
    Emitter.emitSpace();

    rc_recur emitExpression(E.getLhs());

    Emitter.emitSpace();
    Emitter.emitOperator(CEmitter::Symbol::Colon);
    Emitter.emitSpace();

    // The right hand expression does not need parentheses.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::Ternary);

    rc_recur emitExpression(E.getRhs());
  }

  static CEmitter::Symbol getOperator(mlir::Operation *Op) {
    if (mlir::isa<NegOp, SubOp, PtrSubOp, PtrDiffOp>(Op))
      return CEmitter::Symbol::Minus;
    if (mlir::isa<AddOp, PtrAddOp>(Op))
      return CEmitter::Symbol::Plus;
    if (mlir::isa<MulOp, IndirectionOp>(Op))
      return CEmitter::Symbol::Star;
    if (mlir::isa<DivOp>(Op))
      return CEmitter::Symbol::Slash;
    if (mlir::isa<RemOp>(Op))
      return CEmitter::Symbol::Percent;
    if (mlir::isa<LogicalNotOp>(Op))
      return CEmitter::Symbol::Exclaim;
    if (mlir::isa<LogicalAndOp>(Op))
      return CEmitter::Symbol::AmpersandAmpersand;
    if (mlir::isa<LogicalOrOp>(Op))
      return CEmitter::Symbol::PipePipe;
    if (mlir::isa<BitwiseNotOp>(Op))
      return CEmitter::Symbol::Tilde;
    if (mlir::isa<BitwiseAndOp, AddressofOp>(Op))
      return CEmitter::Symbol::Ampersand;
    if (mlir::isa<BitwiseOrOp>(Op))
      return CEmitter::Symbol::Pipe;
    if (mlir::isa<BitwiseXorOp>(Op))
      return CEmitter::Symbol::Caret;
    if (mlir::isa<ShiftLeftOp>(Op))
      return CEmitter::Symbol::LessLess;
    if (mlir::isa<ShiftRightOp>(Op))
      return CEmitter::Symbol::GreaterGreater;
    if (mlir::isa<CmpEqOp>(Op))
      return CEmitter::Symbol::EqualsEquals;
    if (mlir::isa<CmpNeOp>(Op))
      return CEmitter::Symbol::ExclaimEquals;
    if (mlir::isa<CmpLtOp>(Op))
      return CEmitter::Symbol::Less;
    if (mlir::isa<CmpGtOp>(Op))
      return CEmitter::Symbol::Greater;
    if (mlir::isa<CmpLeOp>(Op))
      return CEmitter::Symbol::LessEquals;
    if (mlir::isa<CmpGeOp>(Op))
      return CEmitter::Symbol::GreaterEquals;
    if (mlir::isa<IncrementOp, PostIncrementOp>(Op))
      return CEmitter::Symbol::PlusPlus;
    if (mlir::isa<DecrementOp, PostDecrementOp>(Op))
      return CEmitter::Symbol::MinusMinus;
    if (mlir::isa<AssignOp>(Op))
      return CEmitter::Symbol::Equals;
    if (mlir::isa<CommaOp>(Op))
      return CEmitter::Symbol::Comma;
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
      Emitter.emitPunctuator(CEmitter::Symbol::LeftParenthesis);

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
      Emitter.emitPunctuator(CEmitter::Symbol::RightParenthesis);
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
                               .Attributes = getDeclarationOpAttributes(S),
                               .Kind = CEmitter::EntityKind::LocalVariable,
                             });

    if (not S.getInitializer().empty()) {
      Emitter.emitSpace();
      Emitter.emitOperator(CEmitter::Symbol::Equals);
      Emitter.emitSpace();

      // Comma expressions in a variable initialiser must be parenthesized.
      CurrentPrecedence = OperatorPrecedence::Comma;

      mlir::Value Expression = getExpressionValue(S.getInitializer());

      if (auto Aggregate = Expression.getDefiningOp<AggregateOp>())
        rc_recur emitAggregateInitializer(Aggregate);
      else
        rc_recur emitExpression(Expression);
    }

    Emitter.emitPunctuator(CEmitter::Symbol::Semicolon);
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
    auto Scope = Emitter.enterScope(CEmitter::ScopeKind::None,
                                    CEmitter::Delimiter::None,
                                    /*Indent=*/-1);

    Emitter.emitIdentifier(getNameAttr(S.getLabelOp()),
                           getLocationAttr(S.getLabelOp()),
                           CEmitter::EntityKind::Label,
                           CEmitter::IdentifierKind::Definition);

    Emitter.emitPunctuator(CEmitter::Symbol::Colon);

    if (labelRequiresEmptyExpression(S)) {
      Emitter.emitSpace();
      Emitter.emitPunctuator(CEmitter::Symbol::Semicolon);
    }

    Emitter.emitNewline();

    rc_return;
  }

  RecursiveCoroutine<void> emitExpressionStatement(ExpressionStatementOp S) {
    rc_recur emitExpressionRegion(S.getExpression());
    Emitter.emitPunctuator(CEmitter::Symbol::Semicolon);
    Emitter.emitNewline();
  }

  RecursiveCoroutine<void> emitGotoStatement(GoToOp S) {
    Emitter.emitKeyword(CEmitter::Keyword::Goto);
    Emitter.emitSpace();

    Emitter.emitIdentifier(getNameAttr(S.getLabelOp()),
                           getLocationAttr(S.getLabelOp()),
                           CEmitter::EntityKind::Label,
                           CEmitter::IdentifierKind::Reference);

    Emitter.emitPunctuator(CEmitter::Symbol::Semicolon);
    Emitter.emitNewline();

    rc_return;
  }

  RecursiveCoroutine<void> emitReturnStatement(ReturnOp S) {
    Emitter.emitKeyword(CEmitter::Keyword::Return);

    if (not S.getResult().empty()) {
      Emitter.emitSpace();
      rc_recur emitExpressionRegion(S.getResult());
    }

    Emitter.emitPunctuator(CEmitter::Symbol::Semicolon);
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
      Emitter.emitKeyword(CEmitter::Keyword::If);
      Emitter.emitSpace();
      Emitter.emitPunctuator(CEmitter::Symbol::LeftParenthesis);
      rc_recur emitExpressionRegion(S.getCondition());
      Emitter.emitPunctuator(CEmitter::Symbol::RightParenthesis);

      rc_recur emitImplicitBlockStatement(S.getThen(), EmitBlocks);

      if (S.getElse().empty())
        break;

      if (EmitBlocks)
        Emitter.emitSpace();

      Emitter.emitKeyword(CEmitter::Keyword::Else);

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
      Emitter.emitKeyword(CEmitter::Keyword::Break);
      Emitter.emitPunctuator(CEmitter::Symbol::Semicolon);
      Emitter.emitNewline();
    }
  }

  RecursiveCoroutine<void> emitSwitchStatement(SwitchOp S) {
    Emitter.emitKeyword(CEmitter::Keyword::Switch);
    Emitter.emitSpace();
    Emitter.emitPunctuator(CEmitter::Symbol::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());

    Emitter.emitPunctuator(CEmitter::Symbol::RightParenthesis);
    Emitter.emitSpace();

    // Scope tags are applied within this scope:
    {
      auto Scope = Emitter.enterScope(CEmitter::ScopeKind::BlockStatement,
                                      CEmitter::Delimiter::Braces,
                                      /*Indented=*/false);

      ValueType Type = S.getConditionType();
      for (unsigned I = 0, Count = S.getNumCases(); I < Count; ++I) {
        Emitter.emitKeyword(CEmitter::Keyword::Case);
        Emitter.emitSpace();
        emitIntegerImmediate(S.getCaseValue(I), Type);
        Emitter.emitPunctuator(CEmitter::Symbol::Colon);
        rc_recur emitCaseRegion(S.getCaseRegion(I));
      }

      if (S.hasDefaultCase()) {
        Emitter.emitKeyword(CEmitter::Keyword::Default);
        Emitter.emitPunctuator(CEmitter::Symbol::Colon);
        rc_recur emitCaseRegion(S.getDefaultCaseRegion());
      }
    }

    Emitter.emitNewline();
  }

  RecursiveCoroutine<void> emitForStatement(ForOp S) {
    Emitter.emitKeyword(CEmitter::Keyword::For);
    Emitter.emitSpace();
    Emitter.emitPunctuator(CEmitter::Symbol::LeftParenthesis);
    Emitter.emitPunctuator(CEmitter::Symbol::Semicolon);

    if (not S.getCondition().empty()) {
      Emitter.emitSpace();
      rc_recur emitExpressionRegion(S.getCondition());
    }

    Emitter.emitPunctuator(CEmitter::Symbol::Semicolon);
    if (not S.getExpression().empty()) {
      Emitter.emitSpace();
      rc_recur emitExpressionRegion(S.getExpression());
    }
    Emitter.emitPunctuator(CEmitter::Symbol::RightParenthesis);

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      Emitter.emitNewline();
  }

  RecursiveCoroutine<void> emitWhileStatement(WhileOp S) {
    Emitter.emitKeyword(CEmitter::Keyword::While);
    Emitter.emitSpace();
    Emitter.emitPunctuator(CEmitter::Symbol::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());
    Emitter.emitPunctuator(CEmitter::Symbol::RightParenthesis);

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      Emitter.emitNewline();
  }

  RecursiveCoroutine<void> emitDoWhileStatement(DoWhileOp S) {
    Emitter.emitKeyword(CEmitter::Keyword::Do);

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      Emitter.emitSpace();

    Emitter.emitKeyword(CEmitter::Keyword::While);
    Emitter.emitSpace();
    Emitter.emitPunctuator(CEmitter::Symbol::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());

    Emitter.emitPunctuator(CEmitter::Symbol::RightParenthesis);
    Emitter.emitPunctuator(CEmitter::Symbol::Semicolon);
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
    auto ScopeKind = CEmitter::ScopeKind::None;
    auto Delimiter = CEmitter::Delimiter::None;

    if (EmitBlock) {
      Emitter.emitSpace();
      ScopeKind = CEmitter::ScopeKind::BlockStatement;
      Delimiter = CEmitter::Delimiter::Braces;
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
      auto OuterScope = Emitter
                          .enterScope(CEmitter::ScopeKind::FunctionDeclaration,
                                      CEmitter::Delimiter::None,
                                      /*Indented=*/false);

      llvm::SmallVector<BasicDeclaratorInfo> ParameterDeclarators;
      for (unsigned I = 0; I < Op.getArgCount(); ++I) {
        auto Attrs = Op.getArgAttrs(I);

        auto GetStringAttr = [&](llvm::StringRef Name) {
          return mlir::cast<mlir::StringAttr>(Attrs.get(Name)).getValue();
        };

        mlir::ArrayAttr Attributes = {};
        if (auto Attr = Attrs.get("clift.attributes")) {
          Attributes = mlir::cast<mlir::ArrayAttr>(Attr);
          revng_assert(checkAttributeArray(Attributes));
        }

        ParameterDeclarators.emplace_back(GetStringAttr("clift.name"),
                                          GetStringAttr("clift.handle"),
                                          Attributes);
      }

      rc_recur emitDeclaration(Op.getCliftFunctionType(),
                               DeclaratorInfo{
                                 .Identifier = Op.getName(),
                                 .Location = Op.getHandle(),
                                 .Attributes = getDeclarationOpAttributes(Op),
                                 .Kind = CEmitter::EntityKind::Function,
                                 .Parameters = ParameterDeclarators,
                               });

      Emitter.emitSpace();

      auto InnerScope = Emitter
                          .enterScope(CEmitter::ScopeKind::FunctionDefinition,
                                      CEmitter::Delimiter::Braces);

      Emitter.emitNewline();

      // TODO: Re-enable stack frame inlining.

      rc_recur emitStatementRegion(Op.getBody());

      // TODO: emit a comment containing homeless variable names.
      //       See how old backend does it for reference.
    }

    Emitter.emitNewline();
  }

private:
  CEmitter &Emitter;
  const TargetCImplementation &Target;

  FunctionOp CurrentFunction = {};
  pipeline::Location<decltype(revng::ranks::Function)> CurrentFunctionLocation;

  // Ambient precedence of the current expression.
  OperatorPrecedence CurrentPrecedence = {};
};

} // namespace

void clift::decompile(FunctionOp Function,
                      CEmitter &Emitter,
                      const TargetCImplementation &Target) {
  CliftToCEmitter(Emitter, Target).emitFunction(Function);
}
