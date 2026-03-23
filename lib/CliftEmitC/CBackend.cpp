//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ScopeExit.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftEmitC/CBackend.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/PTML/CTokenEmitter.h"

namespace clift = mlir::clift;
using namespace mlir::clift;

namespace {

static RecursiveCoroutine<void> noopCoroutine() {
  rc_return;
}

static bool hasFallthrough(mlir::Region &R) {
  return not getLastNoFallthroughStatement(R);
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
  // Ambient precedence of the current expression.
  OperatorPrecedence CurrentPrecedence = {};

public:
  using CEmitter::CEmitter;

  static OperatorPrecedence decrementPrecedence(OperatorPrecedence Precedence) {
    revng_assert(Precedence != static_cast<OperatorPrecedence>(0));
    using T = std::underlying_type_t<OperatorPrecedence>;
    return static_cast<OperatorPrecedence>(static_cast<T>(Precedence) - 1);
  }

  void emitCStyleCast(ValueType Type) {
    Tokens.emitOperator(CTE::Operator::LeftParenthesis);
    emitType(Type);
    Tokens.emitOperator(CTE::Operator::RightParenthesis);
    Tokens.emitSpace();
  }

  void
  emitIntegerImmediate(uint64_t Value, ValueType Type, unsigned Radix = 10) {
    Type = dealias(Type, /*IgnoreQualifiers=*/true);

    if (auto Enum = mlir::dyn_cast<EnumType>(Type)) {
      if (auto Enumerator = Enum.getFieldByValue(Value)) {
        Tokens.emitIdentifier(Enumerator.getName(),
                              Enumerator.getHandle(),
                              CTE::EntityKind::Enumerator,
                              CTE::IdentifierKind::Reference);
        return;
      }
    }

    auto Primitive = getUnderlyingIntegerType(Type);

    auto CInteger = Target.getIntegerKind(Primitive.getSize());
    if (not CInteger or *CInteger >= CIntegerKind::Extended
        or not mlir::isa<PrimitiveType>(Type)) {
      // Emit explicit cast if the standard integer type is not known. Emit
      // the literal itself without a suffix (as if int).
      emitCStyleCast(Primitive);
      CInteger = CIntegerKind::Int;
    }

    bool IsSigned = Primitive.getKind() == PrimitiveKind::SignedKind;
    if (IsSigned and static_cast<int64_t>(Value) < 0) {
      Tokens.emitOperator(ptml::CTokenEmitter::Operator::Minus);
      Value = ~Value + 1;
    }

    Tokens.emitIntegerLiteral(llvm::APInt(Primitive.getSize() * 8,
                                          Value,
                                          IsSigned),
                              CTE::IntegerSuffix{ .Unsigned = not IsSigned,
                                                  .MinimumType = *CInteger },
                              Radix);
  }

  //===---------------------------- Expressions ---------------------------===//

  RecursiveCoroutine<void> emitUndefExpression(mlir::Value V) {
    revng_assert(isScalarType(V.getType()));

    Tokens.emitLiteralIdentifier("undef");
    Tokens.emitOperator(CTE::Operator::LeftParenthesis);
    emitType(V.getType());
    Tokens.emitOperator(CTE::Operator::RightParenthesis);

    rc_return;
  }

  RecursiveCoroutine<void> emitImmediateExpression(mlir::Value V) {
    auto E = V.getDefiningOp<ImmediateOp>();

    unsigned Radix = 10;
    if (auto Attr = E->getAttr("clift.radix"))
      Radix = mlir::cast<mlir::IntegerAttr>(Attr).getValue().getZExtValue();

    emitIntegerImmediate(E.getValue(), E.getResult().getType(), Radix);
    rc_return;
  }

  RecursiveCoroutine<void> emitStringLiteralExpression(mlir::Value V) {
    auto E = V.getDefiningOp<StringOp>();
    Tokens.emitStringLiteral(E.getValue());
    rc_return;
  }

  RecursiveCoroutine<void> emitAggregateInitializer(AggregateOp E) {
    // The precedence here must be comma, because an initializer list cannot
    // contain an unparenthesized comma expression. It would be parsed as two
    // initializers instead.
    CurrentPrecedence = OperatorPrecedence::Comma;

    Tokens.emitPunctuator(CTE::Punctuator::LeftBrace);
    for (auto [I, Initializer] : llvm::enumerate(E.getInitializers())) {
      if (I != 0) {
        Tokens.emitPunctuator(CTE::Punctuator::Comma);
        Tokens.emitSpace();
      }

      rc_recur emitExpression(Initializer);
    }
    Tokens.emitPunctuator(CTE::Punctuator::RightBrace);
  }

  RecursiveCoroutine<void> emitAggregateExpression(mlir::Value V) {
    auto E = V.getDefiningOp<AggregateOp>();

    Tokens.emitOperator(CTE::Operator::LeftParenthesis);
    emitType(E.getResult().getType());
    Tokens.emitOperator(CTE::Operator::RightParenthesis);

    rc_recur emitAggregateInitializer(E);
  }

  RecursiveCoroutine<void> emitBlockArgumentExpression(mlir::Value V) {
    auto E = mlir::cast<mlir::BlockArgument>(V);
    mlir::Operation *Op = E.getOwner()->getParentOp();

    if (auto Function = mlir::dyn_cast<FunctionOp>(Op)) {
      const auto &Attrs = Function.getArgAttrs(E.getArgNumber());

      revng_assert(Attrs.getOfType<mlir::StringAttr>("clift.name"),
                   "Function argument name (clift.name) is missing.");

      Tokens.emitIdentifier(Attrs.getString("clift.name"),
                            Attrs.getStringOrEmpty("clift.handle"),
                            CTE::EntityKind::FunctionParameter,
                            CTE::IdentifierKind::Reference);

    } else if (auto For = mlir::dyn_cast<ForOp>(Op)) {
      auto Local = getOnlyOp<LocalVariableOp>(For.getInitializer());
      rc_recur emitLocalVariableExpression(Local.getResult());
    }
  }

  RecursiveCoroutine<void> emitLocalVariableExpression(mlir::Value V) {
    auto E = V.getDefiningOp<LocalVariableOp>();

    Tokens.emitIdentifier(E.getName(),
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

    Tokens.emitIdentifier(Symbol.getName(),
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

    Tokens.emitOperator(E.isIndirect() ? CTE::Operator::Arrow :
                                         CTE::Operator::Dot);

    auto Field = E.getClassType().getFields()[E.getMemberIndex()];

    Tokens.emitIdentifier(Field.getName(),
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

    Tokens.emitOperator(CTE::Operator::LeftBracket);
    rc_recur emitExpression(E.getIndex());
    Tokens.emitOperator(CTE::Operator::RightBracket);
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

    Tokens.emitOperator(CTE::Operator::LeftParenthesis);
    for (auto [I, A] : llvm::enumerate(E.getArguments())) {
      if (I != 0) {
        Tokens.emitPunctuator(CTE::Punctuator::Comma);
        Tokens.emitSpace();
      }

      rc_recur emitExpression(A);
    }
    Tokens.emitOperator(CTE::Operator::RightParenthesis);
  }

  static bool isHiddenCast(CastOpInterface Cast) {
    return mlir::isa<DecayOp>(Cast);
  }

  static mlir::Value unwrapHiddenCasts(CastOpInterface Cast) {
    revng_assert(isHiddenCast(Cast));

    while (true) {
      auto Inner = Cast.getValue().getDefiningOp<CastOpInterface>();
      if (not Inner or not isHiddenCast(Inner))
        break;
    }

    return Cast.getValue();
  }

  static bool requiresExplicitBitCast(BitCastOp Op) {
    auto IsCastableType = [](ValueType T) {
      T = dealias(T, /*IgnoreQualifiers=*/true);

      if (auto P = mlir::dyn_cast<PrimitiveType>(T))
        return isIntegerKind(P.getKind());

      return mlir::isa<EnumType, PointerType>(T);
    };

    return not IsCastableType(Op.getValue().getType())
           or not IsCastableType(Op.getResult().getType());
  }

  RecursiveCoroutine<void> emitBitCastExpression(mlir::Value V) {
    auto E = V.getDefiningOp<BitCastOp>();

    Tokens.emitLiteralIdentifier("bit_cast");
    Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    emitType(E.getResult().getType());
    Tokens.emitPunctuator(CTE::Punctuator::Comma);
    Tokens.emitSpace();

    CurrentPrecedence = OperatorPrecedence::Parentheses;
    rc_recur emitExpression(E.getValue());

    Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);
  }

  RecursiveCoroutine<void> emitCastExpression(mlir::Value V) {
    auto E = V.getDefiningOp<CastOpInterface>();

    emitCStyleCast(E.getResult().getType());

    // Parenthesizing a nested unary prefix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPrefix);

    rc_recur emitExpression(E.getValue());
  }

  RecursiveCoroutine<void> emitHiddenCastExpression(mlir::Value V) {
    auto E = V.getDefiningOp<CastOpInterface>();
    return emitExpression(unwrapHiddenCasts(E));
  }

  RecursiveCoroutine<void> emitTernaryExpression(mlir::Value V) {
    auto E = V.getDefiningOp<TernaryOp>();

    rc_recur emitExpression(E.getCondition());

    Tokens.emitSpace();
    Tokens.emitOperator(CTE::Operator::Question);
    Tokens.emitSpace();

    rc_recur emitExpression(E.getLhs());

    Tokens.emitSpace();
    Tokens.emitOperator(CTE::Operator::Colon);
    Tokens.emitSpace();

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

    Tokens.emitOperator(getOperator(Op));

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
      Tokens.emitSpace();

    // Parenthesizing a nested unary prefix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPrefix);

    return emitExpression(Operand);
  }

  RecursiveCoroutine<void> emitPostfixExpression(mlir::Value V) {
    mlir::Operation *Op = V.getDefiningOp();
    rc_recur emitExpression(Op->getOperand(0));

    // Parenthesizing a nested unary postfix expression is not necessary.
    CurrentPrecedence = decrementPrecedence(OperatorPrecedence::UnaryPostfix);

    Tokens.emitOperator(getOperator(Op));
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
      Tokens.emitSpace();

    Tokens.emitOperator(getOperator(Op));
    Tokens.emitSpace();

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
          .Emit = &CliftToCEmitter::emitBlockArgumentExpression,
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

    if (auto Cast = mlir::dyn_cast<CastOpInterface>(E.getOperation())) {
      if (isHiddenCast(Cast)) {
        auto Info = getExpressionEmitInfo(unwrapHiddenCasts(Cast));

        return {
          .Precedence = decrementPrecedence(Info.Precedence),
          .Emit = &CliftToCEmitter::emitHiddenCastExpression,
        };
      }

      if (auto BitCast = mlir::dyn_cast<BitCastOp>(E.getOperation())) {
        if (requiresExplicitBitCast(BitCast)) {
          return {
            .Precedence = OperatorPrecedence::UnaryPostfix,
            .Emit = &CliftToCEmitter::emitBitCastExpression,
          };
        }
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

  static std::optional<llvm::StringRef> getExpressionLocation(mlir::Value V) {
    if (auto E = V.getDefiningOp<ExpressionOpInterface>()) {
      if (auto NameLoc = mlir::dyn_cast_or_null<mlir::NameLoc>(E->getLoc()))
        return NameLoc.getName();
    }
    return std::nullopt;
  }

  RecursiveCoroutine<void> emitExpression(mlir::Value V) {
    const ExpressionEmitInfo Info = getExpressionEmitInfo(V);

    bool PrintParentheses = Info.Precedence <= CurrentPrecedence
                            and Info.Precedence != OperatorPrecedence::Primary;

    if (PrintParentheses)
      Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    // CurrentPrecedence is changed within this scope:
    {
      const auto PreviousPrecedence = CurrentPrecedence;
      const auto PrecedenceGuard = llvm::make_scope_exit([&]() {
        CurrentPrecedence = PreviousPrecedence;
      });
      CurrentPrecedence = Info.Precedence;

      // If an expression location is available, within this scope an expression
      // region is entered.
      std::optional<CTE::Region> Region;
      if (auto Location = getExpressionLocation(V))
        Region.emplace(Tokens, CTE::RegionKind::Expression, *Location);

      // Emit the expression using the member function returned by
      // getExpressionEmitInfo.
      rc_recur(this->*Info.Emit)(V);
    }

    if (PrintParentheses)
      Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);
  }

  RecursiveCoroutine<void> emitExpressionRegion(mlir::Region &R) {
    mlir::Value Value = getExpressionValue(R);
    revng_assert(Value);
    return emitExpression(Value);
  }

  //===---------------------------- Statements ----------------------------===//

  RecursiveCoroutine<void> emitLocalVariableDeclaration(LocalVariableOp S,
                                                        bool EmitNewline) {
    emitDeclaration(S.getResult().getType(),
                    DeclaratorInfo{
                      .Identifier = S.getName(),
                      .Location = S.getHandle(),
                      .CAttributes = getDeclarationOpCAttributes(S),
                      .Kind = CTE::EntityKind::LocalVariable,
                    });

    if (not S.getInitializer().empty()) {
      Tokens.emitSpace();
      Tokens.emitOperator(CTE::Operator::Equals);
      Tokens.emitSpace();

      // Comma expressions in a variable initialiser must be parenthesized.
      CurrentPrecedence = OperatorPrecedence::Comma;

      mlir::Value Expression = getExpressionValue(S.getInitializer());

      if (auto Aggregate = Expression.getDefiningOp<AggregateOp>())
        rc_recur emitAggregateInitializer(Aggregate);
      else
        rc_recur emitExpression(Expression);
    }

    Tokens.emitPunctuator(CTE::Punctuator::Semicolon);

    if (EmitNewline)
      Tokens.emitNewline();
  }

  bool labelRequiresEmptyExpression(LabelAssignmentOpInterface Op) {
    // Prior to C23, labels cannot be placed at the end of a block:
    if (Op.getOperation() == &Op->getBlock()->back())
      return true;

    // Prior to C23, labels cannot be placed preceding a declaration:
    if (mlir::isa<LocalVariableOp>(&*std::next(Op->getIterator())))
      return true;

    return false;
  }

  void emitLabelStatementImpl(MakeLabelOp Label, bool RequiresEmptyExpression) {
    auto Scope = Tokens.enterScope(CTE::ScopeKind::None,
                                   CTE::Delimiter::None,
                                   /*Indent=*/-1);

    Tokens.emitIdentifier(Label.getName(),
                          Label.getHandle(),
                          CTE::EntityKind::Label,
                          CTE::IdentifierKind::Definition);

    Tokens.emitPunctuator(CTE::Punctuator::Colon);

    if (RequiresEmptyExpression) {
      Tokens.emitSpace();
      Tokens.emitPunctuator(CTE::Punctuator::Semicolon);
    }

    Tokens.emitNewline();
  }

  void emitLabelStatement(MakeLabelOp Label, LabelAssignmentOpInterface Op) {
    emitLabelStatementImpl(Label, labelRequiresEmptyExpression(Op));
  }

  RecursiveCoroutine<void> emitLabelStatement(AssignLabelOp S) {
    emitLabelStatement(S.getLabelOp(), LabelAssignmentOpInterface(S));
    rc_return;
  }

  RecursiveCoroutine<void> emitExpressionStatement(ExpressionStatementOp S) {
    rc_recur emitExpressionRegion(S.getExpression());
    Tokens.emitPunctuator(CTE::Punctuator::Semicolon);
    Tokens.emitNewline();
  }

  RecursiveCoroutine<void>
  emitLabeledJumpStatement(JumpStatementOpInterface S) {
    auto LabelOp = S.getLabel().getDefiningOp<MakeLabelOp>();

    if (mlir::isa<GotoOp>(S))
      Tokens.emitKeyword(CTE::Keyword::Goto);
    else if (mlir::isa<BreakToOp>(S))
      Tokens.emitLiteralIdentifier("break_to");
    else if (mlir::isa<ContinueToOp>(S))
      Tokens.emitLiteralIdentifier("continue_to");
    else
      revng_abort("Unsupported jump statement");

    Tokens.emitSpace();
    Tokens.emitIdentifier(LabelOp.getName(),
                          LabelOp.getHandle(),
                          CTE::EntityKind::Label,
                          CTE::IdentifierKind::Reference);

    Tokens.emitPunctuator(CTE::Punctuator::Semicolon);
    Tokens.emitNewline();

    rc_return;
  }

  RecursiveCoroutine<void> emitReturnStatement(ReturnOp S) {
    Tokens.emitKeyword(CTE::Keyword::Return);

    if (not S.getResult().empty()) {
      Tokens.emitSpace();
      rc_recur emitExpressionRegion(S.getResult());
    }

    Tokens.emitPunctuator(CTE::Punctuator::Semicolon);
    Tokens.emitNewline();
  }

  static bool mayElideIfStatementBraces(IfOp If) {
    while (true) {
      if (not mayElideBraces(If.getThen()))
        return false;

      if (If.getElse().empty())
        return true;

      auto ElseIf = getOnlyOp<IfOp>(If.getElse());

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
      Tokens.emitKeyword(CTE::Keyword::If);
      Tokens.emitSpace();
      Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);
      rc_recur emitExpressionRegion(S.getCondition());
      Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);

      rc_recur emitImplicitBlockStatement(S.getThen(), EmitBlocks);

      if (S.getElse().empty())
        break;

      if (EmitBlocks)
        Tokens.emitSpace();

      Tokens.emitKeyword(CTE::Keyword::Else);

      if (auto ElseIf = getOnlyOp<IfOp>(S.getElse())) {
        S = ElseIf;
        Tokens.emitSpace();
      } else {
        rc_recur emitImplicitBlockStatement(S.getElse(), EmitBlocks);
        break;
      }
    }

    if (EmitBlocks)
      Tokens.emitNewline();
  }

  RecursiveCoroutine<void> emitCaseRegion(mlir::Region &R) {
    bool Break = hasFallthrough(R);

    if (rc_recur emitImplicitBlockStatement(R)) {
      if (Break)
        Tokens.emitSpace();
      else
        Tokens.emitNewline();
    }

    if (Break) {
      Tokens.emitKeyword(CTE::Keyword::Break);
      Tokens.emitPunctuator(CTE::Punctuator::Semicolon);
      Tokens.emitNewline();
    }
  }

  RecursiveCoroutine<void> emitSwitchStatement(SwitchOp S) {
    unsigned Radix = 10;
    if (auto Attr = S->getAttr("clift.radix"))
      Radix = mlir::cast<mlir::IntegerAttr>(Attr).getValue().getZExtValue();

    Tokens.emitKeyword(CTE::Keyword::Switch);
    Tokens.emitSpace();
    Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());

    Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);
    Tokens.emitSpace();

    // Scope tags are applied within this scope:
    {
      auto Scope = Tokens.enterScope(CTE::ScopeKind::BlockStatement,
                                     CTE::Delimiter::Braces,
                                     /*Indented=*/false);

      Tokens.emitNewline();

      ValueType Type = S.getConditionType();
      for (unsigned I = 0, Count = S.getNumCases(); I < Count; ++I) {
        Tokens.emitKeyword(CTE::Keyword::Case);
        Tokens.emitSpace();
        emitIntegerImmediate(S.getCaseValue(I), Type, Radix);
        Tokens.emitPunctuator(CTE::Punctuator::Colon);
        rc_recur emitCaseRegion(S.getCaseRegion(I));
      }

      if (S.hasDefaultCase()) {
        Tokens.emitKeyword(CTE::Keyword::Default);
        Tokens.emitPunctuator(CTE::Punctuator::Colon);
        rc_recur emitCaseRegion(S.getDefaultCaseRegion());
      }
    }

    Tokens.emitNewline();
  }

  RecursiveCoroutine<bool> emitLoopBodyWithContinueLabel(mlir::Region &Region,
                                                         MakeLabelOp Continue) {
    auto Emit = [this, Continue](mlir::Region &R) -> RecursiveCoroutine<void> {
      rc_recur emitStatementRegion(R);
      emitLabelStatementImpl(Continue, /*RequiresEmptyExpression=*/true);
    };

    return emitImplicitBlockStatement(Region, true, Emit);
  }

  RecursiveCoroutine<bool> emitLoopBody(LoopOpInterface Loop,
                                        mlir::Region &Region) {
    if (auto Continue = Loop.getContinueLabel()) {
      auto Label = Continue.getDefiningOp<MakeLabelOp>();
      return emitLoopBodyWithContinueLabel(Region, Label);
    }

    return emitImplicitBlockStatement(Region);
  }

  RecursiveCoroutine<void> emitForStatement(ForOp S) {
    Tokens.emitKeyword(CTE::Keyword::For);
    Tokens.emitSpace();
    Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    if (mlir::Region &R = S.getInitializer(); not R.empty()) {
      mlir::Operation *Op = getOnlyOp(R);
      if (auto L = mlir::dyn_cast<LocalVariableOp>(Op))
        rc_recur emitLocalVariableDeclaration(L, /*Newline=*/false);
      else
        rc_recur emitExpressionStatement(mlir::cast<ExpressionStatementOp>(Op));
    } else {
      Tokens.emitPunctuator(CTE::Punctuator::Semicolon);
    }

    if (mlir::Region &R = S.getCondition(); not R.empty()) {
      Tokens.emitSpace();
      rc_recur emitExpressionRegion(R);
    }
    Tokens.emitPunctuator(CTE::Punctuator::Semicolon);

    if (mlir::Region &R = S.getExpression(); not R.empty()) {
      Tokens.emitSpace();
      rc_recur emitExpressionRegion(R);
    }
    Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);

    if (rc_recur emitLoopBody(S, S.getBody()))
      Tokens.emitNewline();

    if (auto Break = S.getBreakLabel())
      emitLabelStatement(Break.getDefiningOp<MakeLabelOp>(), S);
  }

  RecursiveCoroutine<void> emitWhileStatement(WhileOp S) {
    Tokens.emitKeyword(CTE::Keyword::While);
    Tokens.emitSpace();
    Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());
    Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);

    if (rc_recur emitLoopBody(S, S.getBody()))
      Tokens.emitNewline();

    if (auto Break = S.getBreakLabel())
      emitLabelStatement(Break.getDefiningOp<MakeLabelOp>(), S);
  }

  RecursiveCoroutine<void> emitDoWhileStatement(DoWhileOp S) {
    Tokens.emitKeyword(CTE::Keyword::Do);

    if (rc_recur emitLoopBody(S, S.getBody()))
      Tokens.emitSpace();

    Tokens.emitKeyword(CTE::Keyword::While);
    Tokens.emitSpace();
    Tokens.emitPunctuator(CTE::Punctuator::LeftParenthesis);

    rc_recur emitExpressionRegion(S.getCondition());

    Tokens.emitPunctuator(CTE::Punctuator::RightParenthesis);
    Tokens.emitPunctuator(CTE::Punctuator::Semicolon);
    Tokens.emitNewline();

    if (auto Break = S.getBreakLabel())
      emitLabelStatement(Break.getDefiningOp<MakeLabelOp>(), S);
  }

  RecursiveCoroutine<void> emitBlockStatement(BlockStatementOp S) {
    {
      auto Scope = Tokens.enterScope(CTE::ScopeKind::BlockStatement,
                                     CTE::Delimiter::Braces);

      Tokens.emitNewline();
      rc_recur emitStatementRegion(S.getBlock());
    }
    Tokens.emitNewline();
  }

  static mlir::ArrayAttr getComments(StatementOpInterface S) {
    return mlir::cast_or_null<mlir::ArrayAttr>(S->getAttr("clift.comments"));
  }

  RecursiveCoroutine<void> emitStatement(StatementOpInterface Stmt) {
    mlir::Operation *Op = Stmt.getOperation();

    if (auto Comments = getComments(Stmt)) {
      // TODO: Add a comment formatting layer on top of CE.
      //       At least spaces at the start of each line would be nice.
      auto CE = Tokens.emitComment(CTE::CommentKind::Line);

      for (mlir::Attribute CommentAttr : Comments) {
        CE.emit(mlir::cast<mlir::StringAttr>(CommentAttr).getValue());
        CE.emit("\n");
      }
    }

    if (auto S = mlir::dyn_cast<LocalVariableOp>(Op))
      return emitLocalVariableDeclaration(S, /*Newline=*/true);

    if (auto S = mlir::dyn_cast<MakeLabelOp>(Op))
      return noopCoroutine();

    if (auto S = mlir::dyn_cast<AssignLabelOp>(Op))
      return emitLabelStatement(S);

    if (auto S = mlir::dyn_cast<ExpressionStatementOp>(Op))
      return emitExpressionStatement(S);

    if (auto S = mlir::dyn_cast<JumpStatementOpInterface>(Op))
      return emitLabeledJumpStatement(S);

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

    if (auto S = mlir::dyn_cast<BlockStatementOp>(Op))
      return emitBlockStatement(S);

    revng_abort("Unsupported operation");
  }

  RecursiveCoroutine<void> emitStatementRegion(mlir::Region &R) {
    for (mlir::Operation &Stmt : R.getOps())
      rc_recur emitStatement(mlir::cast<StatementOpInterface>(&Stmt));
  }

  static bool mayElideBraces(mlir::Operation *Operation) {
    return mlir::isa<ExpressionStatementOp,
                     GotoOp,
                     ReturnOp,
                     BreakToOp,
                     ContinueToOp>(Operation);
  }

  static bool mayElideBraces(mlir::Region &R) {
    mlir::Operation *OnlyOp = getOnlyOp(R);
    return OnlyOp != nullptr and mayElideBraces(OnlyOp);
  }

  RecursiveCoroutine<bool>
  emitImplicitBlockStatement(mlir::Region &R, bool EmitBlock, auto EmitRegion) {
    auto ScopeKind = CTE::ScopeKind::None;
    auto Delimiter = CTE::Delimiter::None;

    if (EmitBlock) {
      Tokens.emitSpace();
      ScopeKind = CTE::ScopeKind::BlockStatement;
      Delimiter = CTE::Delimiter::Braces;
    }

    auto Scope = Tokens.enterScope(ScopeKind, Delimiter);
    Tokens.emitNewline();

    auto ExplicitBlock = getOnlyOp<BlockStatementOp>(R);
    rc_recur EmitRegion(ExplicitBlock ? ExplicitBlock.getBlock() : R);

    rc_return EmitBlock;
  }

  RecursiveCoroutine<bool> emitImplicitBlockStatement(mlir::Region &R,
                                                      bool EmitBlock) {
    return emitImplicitBlockStatement(R, EmitBlock, [this](mlir::Region &R) {
      return emitStatementRegion(R);
    });
  }

  RecursiveCoroutine<bool> emitImplicitBlockStatement(mlir::Region &R) {
    return emitImplicitBlockStatement(R, not mayElideBraces(R));
  }

  //===----------------------------- Functions ----------------------------===//

  RecursiveCoroutine<void> emitFunction(FunctionOp Op) {
    // Scope tags are applied within this scope:
    {
      auto OuterScope = Tokens.enterScope(CTE::ScopeKind::FunctionDeclaration,
                                          CTE::Delimiter::None,
                                          /*Indented=*/false);

      emitFunctionPrototype(Op);

      Tokens.emitSpace();

      auto InnerScope = Tokens.enterScope(CTE::ScopeKind::FunctionDefinition,
                                          CTE::Delimiter::Braces);

      Tokens.emitNewline();

      // TODO: Re-enable stack frame inlining.

      rc_recur emitStatementRegion(Op.getBody());

      // TODO: emit a comment containing homeless variable names.
      //       See how old backend does it for reference.
    }

    Tokens.emitNewline();
  }
};

} // namespace

void clift::decompile(FunctionOp Function,
                      ptml::CTokenEmitter &Emitter,
                      const TargetCImplementation &Target) {
  CliftToCEmitter(Emitter, Target).emitFunction(Function);
}
