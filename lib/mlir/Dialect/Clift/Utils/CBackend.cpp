//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ScopeExit.h"

#include "revng/mlir/Dialect/Clift/Utils/CBackend.h"
#include "revng/mlir/Dialect/Clift/Utils/CEmitter.h"

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

class CliftToCEmitter : CEmitter {
  FunctionOp CurrentFunction = {};

  // Ambient precedence of the current expression.
  OperatorPrecedence CurrentPrecedence = {};

public:
  using CEmitter::CEmitter;

  llvm::StringRef getStringAttr(mlir::Operation *Op, llvm::StringRef Name) {
    return mlir::cast<mlir::StringAttr>(Op->getAttr(Name)).getValue();
  }

  llvm::StringRef getNameAttr(mlir::Operation *Op) {
    return getStringAttr(Op, "clift.name");
  }

  llvm::StringRef getLocationAttr(mlir::Operation *Op) {
    return getStringAttr(Op, "clift.handle");
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

        C.emitOperator(CTE::Operator::LeftParenthesis);
        emitPrimitiveType(T);
        C.emitOperator(CTE::Operator::RightParenthesis);

        Integer = CIntegerKind::Int;
      }

      C.emitIntegerLiteral(makeIntegerValue(T, Value), *Integer, 10);
    } else {
      auto Enum = mlir::cast<EnumType>(Type);
      auto Enumerator = Enum.getFieldByValue(Value);

      C.emitIdentifier(Enumerator.getName(),
                       Enumerator.getHandle(),
                       CTE::EntityKind::Enumerator,
                       CTE::IdentifierKind::Reference);
    }
  }

  //===---------------------------- Expressions ---------------------------===//

  RecursiveCoroutine<void> emitUndefExpression(mlir::Value V) {
    revng_assert(isScalarType(V.getType()));

    C.emitLiteralIdentifier("undef");
    C.emitOperator(CTE::Operator::LeftParenthesis);
    emitType(V.getType());
    C.emitOperator(CTE::Operator::RightParenthesis);

    rc_return;
  }

  RecursiveCoroutine<void> emitImmediateExpression(mlir::Value V) {
    auto E = V.getDefiningOp<ImmediateOp>();
    emitIntegerImmediate(E.getValue(), E.getResult().getType());
    rc_return;
  }

  RecursiveCoroutine<void> emitStringLiteralExpression(mlir::Value V) {
    auto E = V.getDefiningOp<StringOp>();
    C.emitStringLiteral(E.getValue());
    rc_return;
  }

  RecursiveCoroutine<void> emitAggregateInitializer(AggregateOp E) {
    // The precedence here must be comma, because an initializer list cannot
    // contain an unparenthesized comma expression. It would be parsed as two
    // initializers instead.
    CurrentPrecedence = OperatorPrecedence::Comma;

    C.emitPunctuator(CTE::Punctuator::LeftBrace);
    for (auto [I, Initializer] : llvm::enumerate(E.getInitializers())) {
      if (I != 0) {
        C.emitPunctuator(CTE::Punctuator::Comma);
        C.emitSpace();
      }

      rc_recur emitExpression(Initializer);
    }
    C.emitPunctuator(CTE::Punctuator::RightBrace);
  }

  RecursiveCoroutine<void> emitAggregateExpression(mlir::Value V) {
    auto E = V.getDefiningOp<AggregateOp>();

    C.emitOperator(CTE::Operator::LeftParenthesis);
    emitType(E.getResult().getType());
    C.emitOperator(CTE::Operator::RightParenthesis);

    rc_recur emitAggregateInitializer(E);
  }

  RecursiveCoroutine<void> emitParameterExpression(mlir::Value V) {
    auto E = mlir::cast<mlir::BlockArgument>(V);

    const auto &ArgAttrs = CurrentFunction.getArgAttrs(E.getArgNumber());
    const auto GetStringAttr = [&ArgAttrs](llvm::StringRef Name) {
      return mlir::cast<mlir::StringAttr>(ArgAttrs.get(Name)).getValue();
    };

    C.emitIdentifier(GetStringAttr("clift.name"),
                     GetStringAttr("clift.handle"),
                     CTE::EntityKind::FunctionParameter,
                     CTE::IdentifierKind::Reference);
    rc_return;
  }

  RecursiveCoroutine<void> emitLocalVariableExpression(mlir::Value V) {
    auto E = V.getDefiningOp<LocalVariableOp>();

    C.emitIdentifier(getNameAttr(E),
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

    C.emitIdentifier(Symbol.getName(),
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

    C.emitOperator(E.isIndirect() ? CTE::Operator::Arrow : CTE::Operator::Dot);

    auto Field = E.getClassType().getFields()[E.getMemberIndex()];

    C.emitIdentifier(Field.getName(),
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

    C.emitOperator(CTE::Operator::LeftBracket);
    rc_recur emitExpression(E.getIndex());
    C.emitOperator(CTE::Operator::RightBracket);
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

    C.emitOperator(CTE::Operator::LeftParenthesis);
    for (auto [I, A] : llvm::enumerate(E.getArguments())) {
      if (I != 0) {
        C.emitPunctuator(CTE::Punctuator::Comma);
        C.emitSpace();
      }

      rc_recur emitExpression(A);
    }
    C.emitOperator(CTE::Operator::RightParenthesis);
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

    C.emitOperator(CTE::Operator::LeftParenthesis);
    emitType(E.getResult().getType());
    C.emitOperator(CTE::Operator::RightParenthesis);

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

    C.emitSpace();
    C.emitOperator(CTE::Operator::Question);
    C.emitSpace();

    rc_recur emitExpression(E.getLhs());

    C.emitSpace();
    C.emitOperator(CTE::Operator::Colon);
    C.emitSpace();

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

    C.emitOperator(getOperator(Op));

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
      C.emitSpace();

    // Parenthesizing a nested unary prefix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPrefix);

    return emitExpression(Operand);
  }

  RecursiveCoroutine<void> emitPostfixExpression(mlir::Value V) {
    mlir::Operation *Op = V.getDefiningOp();
    rc_recur emitExpression(Op->getOperand(0));

    // Parenthesizing a nested unary postfix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);

    C.emitOperator(getOperator(Op));
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
      C.emitSpace();

    C.emitOperator(getOperator(Op));
    C.emitSpace();

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
      C.emitPunctuator(CTE::Punctuator::LeftParenthesis);

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
      C.emitPunctuator(CTE::Punctuator::RightParenthesis);
  }

  RecursiveCoroutine<void> emitExpressionRegion(mlir::Region &R) {
    mlir::Value Value = getExpressionValue(R);
    revng_assert(Value);
    return emitExpression(Value);
  }

  //===---------------------------- Statements ----------------------------===//

  RecursiveCoroutine<void> emitLocalVariableDeclaration(LocalVariableOp S) {
    emitDeclaration(S.getResult().getType(),
                    DeclaratorInfo{
                      .Identifier = getNameAttr(S),
                      .Location = S.getHandle(),
                      .Attributes = getDeclarationOpAttributes(S),
                      .Kind = CTE::EntityKind::LocalVariable,
                    });

    if (not S.getInitializer().empty()) {
      C.emitSpace();
      C.emitOperator(CTE::Operator::Equals);
      C.emitSpace();

      // Comma expressions in a variable initialiser must be parenthesized.
      CurrentPrecedence = OperatorPrecedence::Comma;

      mlir::Value Expression = getExpressionValue(S.getInitializer());

      if (auto Aggregate = Expression.getDefiningOp<AggregateOp>())
        rc_recur emitAggregateInitializer(Aggregate);
      else
        rc_recur emitExpression(Expression);
    }

    C.emitPunctuator(CTE::Punctuator::Semicolon);
    C.emitNewline();
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
    auto Scope = C.enterScope(CTE::ScopeKind::None,
                              CTE::Delimiter::None,
                              /*Indent=*/-1);

    C.emitIdentifier(getNameAttr(S.getLabelOp()),
                     getLocationAttr(S.getLabelOp()),
                     CTE::EntityKind::Label,
                     CTE::IdentifierKind::Definition);

    C.emitPunctuator(CTE::Punctuator::Colon);

    if (labelRequiresEmptyExpression(S)) {
      C.emitSpace();
      C.emitPunctuator(CTE::Punctuator::Semicolon);
    }

    C.emitNewline();

    rc_return;
  }

  RecursiveCoroutine<void> emitExpressionStatement(ExpressionStatementOp S) {
    rc_recur emitExpressionRegion(S.getExpression());
    C.emitPunctuator(CTE::Punctuator::Semicolon);
    C.emitNewline();
  }

  RecursiveCoroutine<void> emitGotoStatement(GoToOp S) {
    C.emitKeyword(CTE::Keyword::Goto);
    C.emitSpace();

    C.emitIdentifier(getNameAttr(S.getLabelOp()),
                     getLocationAttr(S.getLabelOp()),
                     CTE::EntityKind::Label,
                     CTE::IdentifierKind::Reference);

    C.emitPunctuator(CTE::Punctuator::Semicolon);
    C.emitNewline();

    rc_return;
  }

  RecursiveCoroutine<void> emitReturnStatement(ReturnOp S) {
    C.emitKeyword(CTE::Keyword::Return);

    if (not S.getResult().empty()) {
      C.emitSpace();
      rc_recur emitExpressionRegion(S.getResult());
    }

    C.emitPunctuator(CTE::Punctuator::Semicolon);
    C.emitNewline();
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
      C.emitKeyword(CTE::Keyword::If);
      C.emitSpace();
      C.emitPunctuator(CTE::Punctuator::LeftParenthesis);
      rc_recur emitExpressionRegion(S.getCondition());
      C.emitPunctuator(CTE::Punctuator::RightParenthesis);

      rc_recur emitImplicitBlockStatement(S.getThen(), EmitBlocks);

      if (S.getElse().empty())
        break;

      if (EmitBlocks)
        C.emitSpace();

      C.emitKeyword(CTE::Keyword::Else);

      if (auto ElseIf = getOnlyOperation<IfOp>(S.getElse())) {
        S = ElseIf;
        C.emitSpace();
      } else {
        rc_recur emitImplicitBlockStatement(S.getElse(), EmitBlocks);
        break;
      }
    }

    if (EmitBlocks)
      C.emitNewline();
  }

  RecursiveCoroutine<void> emitCaseRegion(mlir::Region &R) {
    bool Break = hasFallthrough(R);

    if (rc_recur emitImplicitBlockStatement(R)) {
      if (Break)
        C.emitSpace();
      else
        C.emitNewline();
    }

    if (Break) {
      C.emitKeyword(CTE::Keyword::Break);
      C.emitPunctuator(CTE::Punctuator::Semicolon);
      C.emitNewline();
    }
  }

  RecursiveCoroutine<void> emitSwitchStatement(SwitchOp S) {
    C.emitKeyword(CTE::Keyword::Switch);
    C.emitSpace();
    C.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());

    C.emitPunctuator(CTE::Punctuator::RightParenthesis);
    C.emitSpace();

    // Scope tags are applied within this scope:
    {
      auto Scope = C.enterScope(CTE::ScopeKind::BlockStatement,
                                CTE::Delimiter::Braces,
                                /*Indented=*/false);

      ValueType Type = S.getConditionType();
      for (unsigned I = 0, Count = S.getNumCases(); I < Count; ++I) {
        C.emitKeyword(CTE::Keyword::Case);
        C.emitSpace();
        emitIntegerImmediate(S.getCaseValue(I), Type);
        C.emitPunctuator(CTE::Punctuator::Colon);
        rc_recur emitCaseRegion(S.getCaseRegion(I));
      }

      if (S.hasDefaultCase()) {
        C.emitKeyword(CTE::Keyword::Default);
        C.emitPunctuator(CTE::Punctuator::Colon);
        rc_recur emitCaseRegion(S.getDefaultCaseRegion());
      }
    }

    C.emitNewline();
  }

  RecursiveCoroutine<void> emitForStatement(ForOp S) {
    C.emitKeyword(CTE::Keyword::For);
    C.emitSpace();
    C.emitPunctuator(CTE::Punctuator::LeftParenthesis);
    C.emitPunctuator(CTE::Punctuator::Semicolon);

    if (not S.getCondition().empty()) {
      C.emitSpace();
      rc_recur emitExpressionRegion(S.getCondition());
    }

    C.emitPunctuator(CTE::Punctuator::Semicolon);
    if (not S.getExpression().empty()) {
      C.emitSpace();
      rc_recur emitExpressionRegion(S.getExpression());
    }
    C.emitPunctuator(CTE::Punctuator::RightParenthesis);

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      C.emitNewline();
  }

  RecursiveCoroutine<void> emitWhileStatement(WhileOp S) {
    C.emitKeyword(CTE::Keyword::While);
    C.emitSpace();
    C.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());
    C.emitPunctuator(CTE::Punctuator::RightParenthesis);

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      C.emitNewline();
  }

  RecursiveCoroutine<void> emitDoWhileStatement(DoWhileOp S) {
    C.emitKeyword(CTE::Keyword::Do);

    if (rc_recur emitImplicitBlockStatement(S.getBody()))
      C.emitSpace();

    C.emitKeyword(CTE::Keyword::While);
    C.emitSpace();
    C.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());

    C.emitPunctuator(CTE::Punctuator::RightParenthesis);
    C.emitPunctuator(CTE::Punctuator::Semicolon);
    C.emitNewline();
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
      C.emitSpace();
      ScopeKind = CTE::ScopeKind::BlockStatement;
      Delimiter = CTE::Delimiter::Braces;
    }

    auto Scope = C.enterScope(ScopeKind, Delimiter);
    C.emitNewline();

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
      auto OuterScope = C.enterScope(CTE::ScopeKind::FunctionDeclaration,
                                     CTE::Delimiter::None,
                                     /*Indented=*/false);

      llvm::SmallVector<ParameterDeclaratorInfo> ParameterDeclarators;
      for (unsigned I = 0; I < Op.getArgCount(); ++I) {
        auto Attrs = Op.getArgAttrs(I);

        auto GetStringAttr = [&Attrs](llvm::StringRef Name) {
          return mlir::cast<mlir::StringAttr>(Attrs.get(Name)).getValue();
        };

        mlir::ArrayAttr Attributes = {};
        if (auto Attr = Attrs.get("clift.attributes")) {
          Attributes = mlir::cast<mlir::ArrayAttr>(Attr);
          revng_assert(isValidAttributeArray(Attributes));
        }

        ParameterDeclarators.emplace_back(GetStringAttr("clift.name"),
                                          GetStringAttr("clift.handle"),
                                          Attributes);
      }

      emitDeclaration(Op.getCliftFunctionType(),
                      DeclaratorInfo{
                        .Identifier = Op.getName(),
                        .Location = Op.getHandle(),
                        .Attributes = getDeclarationOpAttributes(Op),
                        .Kind = CTE::EntityKind::Function,
                        .Parameters = ParameterDeclarators,
                      });

      C.emitSpace();

      auto InnerScope = C.enterScope(CTE::ScopeKind::FunctionDefinition,
                                     CTE::Delimiter::Braces);

      C.emitNewline();

      // TODO: Re-enable stack frame inlining.

      rc_recur emitStatementRegion(Op.getBody());

      // TODO: emit a comment containing homeless variable names.
      //       See how old backend does it for reference.
    }

    C.emitNewline();
  }
};

} // namespace

void clift::decompile(FunctionOp Function,
                      CTokenEmitter &Emitter,
                      const TargetCImplementation &Target) {
  CliftToCEmitter(Emitter, Target).emitFunction(Function);
}
