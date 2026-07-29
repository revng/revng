//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Clift/CliftOpHelpers.h"
#include "revng/CliftEmitC/CBackend.h"
#include "revng/CliftEmitC/CEmitter.h"
#include "revng/CliftEmitC/TypeDefinitionEmitter.h"
#include "revng/PTML/CTokenEmitter.h"
#include "revng/Ranks/Location.h"
#include "revng/Ranks/Ranks.h"

using namespace clift;

namespace {

inline bool isVisibleStatement(mlir::Operation *Op) {
  return not mlir::isa<MakeLabelOp, RequireOp>(Op);
}

inline auto getVisibleStatementRange(mlir::Region &R) {
  return llvm::make_filter_range(R.getOps(), [](mlir::Operation &Op) {
    return isVisibleStatement(&Op);
  });
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

  // Configuration controlling optional emission behaviour for this function,
  // most notably whether the stack-frame struct definition should be inlined
  // at the top of the function body.
  CBackendConfiguration Configuration;

public:
  CliftToCEmitter(ptml::CTokenEmitter &Emitter,
                  const CDataModel &DataModel,
                  CBackendConfiguration Configuration) :
    CEmitter(Emitter, DataModel), Configuration(Configuration) {}

  using CEmitter::CEmitter;

  static OperatorPrecedence decrementPrecedence(OperatorPrecedence Precedence) {
    revng_assert(Precedence != static_cast<OperatorPrecedence>(0));
    using T = std::underlying_type_t<OperatorPrecedence>;
    return static_cast<OperatorPrecedence>(static_cast<T>(Precedence) - 1);
  }

  void emitCStyleCast(mlir::Type Type) {
    Tokens.emitOperator(CTE::Operator::LeftParenthesis);
    emitType(Type);
    Tokens.emitOperator(CTE::Operator::RightParenthesis);
    Tokens.emitSpace();
  }

  void emitEnumImmediate(uint64_t Value, EnumType Type) {
    auto Enumerator = Type.getFieldByValue(Value);
    revng_assert(Enumerator);

    Tokens.emitIdentifier(Enumerator.getName(),
                          Enumerator.getHandle(),
                          CTE::EntityKind::Enumerator,
                          CTE::IdentifierKind::Reference);
  }

  void emitIntegerLiteral(uint64_t Value,
                          bool IsSigned,
                          CStandardType Type,
                          unsigned Radix) {
    if (IsSigned and static_cast<int64_t>(Value) < 0) {
      Tokens.emitOperator(ptml::CTokenEmitter::Operator::Minus);
      Value = ~Value + 1;
    }

    Tokens.emitIntegerLiteral(llvm::APInt(64, Value, IsSigned),
                              CTE::IntegerSuffix{ .Unsigned = not IsSigned,
                                                  .MinimumType = Type },
                              Radix);
  }

  //===---------------------------- Expressions ---------------------------===//

  RecursiveCoroutine<void> emitUndefExpression(mlir::Value V) {
    Tokens.emitMacro("undef");
    Tokens.emitOperator(CTE::Operator::LeftParenthesis);
    emitType(V.getType());
    Tokens.emitOperator(CTE::Operator::RightParenthesis);

    rc_return;
  }

  static bool isNullPointerConstant(ImmediateOp E) {
    if (E.getValue() != 0)
      return false;

    if (auto Cast = getOnlyUser<BitCastOp>(E))
      return clift::unwrapped_isa<PointerType>(Cast.getResult().getType());

    return false;
  }

  RecursiveCoroutine<void> emitImmediateExpression(mlir::Value V) {
    auto E = V.getDefiningOp<ImmediateOp>();
    mlir::Type Type = unwrapTypedefs(E.getType());

    if (auto Enum = mlir::dyn_cast<EnumType>(Type))
      rc_return emitEnumImmediate(E.getValue(), Enum);

    unsigned Radix = 10;
    if (auto Attr = E->getAttr("clift.radix"))
      Radix = mlir::cast<mlir::IntegerAttr>(Attr).getValue().getZExtValue();

    auto IntType = mlir::cast<IntegerType>(Type);
    auto CType = CStandardType::Int;

    // Using any specific integer suffix is only required when the value is not
    // immediately converted to another integer type. While such casts are
    // usually removed by expression rewriting, some may be reintroduced during
    // legalization.
    auto Cast = getOnlyUser<CastOpInterface>(V);
    if (not Cast
        or not unwrapped_isa<IntegralType>(Cast.getResult().getType())) {
      auto Range = DataModel.getStandardIntegerRange(IntType.getSize());
      revng_assert(Range, "Integer immediate not representable in C.");
      CType = Range->first;
    }

    emitIntegerLiteral(E.getValue(), IntType.isSigned(), CType, Radix);
    rc_return;
  }

  RecursiveCoroutine<void> emitNullPointerConstant(mlir::Value V) {
    Tokens.emitMacro("NULL");
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

    } else if (auto S = mlir::dyn_cast<StatementOpInterface>(Op)) {
      rc_recur emitLocalVariableExpression(S.getBlockArgumentVariable(E));
    } else {
      revng_abort("Unsupported block argument.");
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
    return mlir::isa<DecayOp>(Cast) or Cast->hasAttr("clift.implicit");
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
    auto IsCastableType = [](mlir::Type T) {
      return clift::unwrapped_isa<IntegerType, EnumType, PointerType>(T);
    };

    return not IsCastableType(Op.getValue().getType())
           or not IsCastableType(Op.getResult().getType());
  }

  RecursiveCoroutine<void> emitBitCastExpression(mlir::Value V) {
    auto E = V.getDefiningOp<BitCastOp>();

    Tokens.emitMacro("bit_cast");
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
    CurrentPrecedence = decrementPrecedence(CurrentPrecedence);
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
        if (auto T = mlir::dyn_cast<IntegerType>(I.getResult().getType()))
          return T.isSigned() and static_cast<int64_t>(I.getValue()) < 0;
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

    if (auto Immediate = mlir::dyn_cast<ImmediateOp>(E.getOperation())) {
      if (isNullPointerConstant(Immediate)) {
        return {
          .Precedence = OperatorPrecedence::Primary,
          .Emit = &CliftToCEmitter::emitNullPointerConstant,
        };
      }

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
          .Precedence = Info.Precedence,
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

  RecursiveCoroutine<void> emitLocalVariableDeclaration(LocalVariableOp Var,
                                                        bool EmitNewline) {
    mlir::Type Type = Var.getResult().getType();
    DeclaratorInfo Declarator{
      .Identifier = Var.getName(),
      .Location = Var.getHandle(),
      .CAttributes = getDeclarationOpCAttributes(Var),
      .Kind = CTE::EntityKind::LocalVariable,
    };

    bool DefinitionEmitted = false;
    if (Configuration.InlineStackFrameType) {
      // When the configuration enables stack-frame inlining,
      if (auto Flag = Var->getAttrOfType<mlir::BoolAttr>("clift.stack_frame")) {
        if (Flag.getValue()) {
          // and the variable has been tagged as the canonical stack frame,
          if (auto S = clift::unwrapped_dyn_cast<clift::StructType>(Type)) {
            // emit the *definition* of the struct/union type inline instead of
            // a regular declaration, so the C output reads as
            // `struct _PACKED ... my_stack { ... } var_1;`.
            //
            // Note that this skips all the typedefs that might be there before
            // the struct.
            TypeDefinitionEmitter Emitter(Tokens,
                                          DataModel,
                                          Configuration.TypeEmitter);
            Emitter.emitInlineClassDefinition(S, std::move(Declarator));
            DefinitionEmitted = true;
          }
        }
      }
    }

    if (not DefinitionEmitted)
      emitDeclaration(Type, Declarator);

    if (not Var.getInitializer().empty()) {
      Tokens.emitSpace();
      Tokens.emitOperator(CTE::Operator::Equals);
      Tokens.emitSpace();

      // Comma expressions in a variable initialiser must be parenthesized.
      CurrentPrecedence = OperatorPrecedence::Comma;

      mlir::Value Expression = getExpressionValue(Var.getInitializer());

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
    mlir::Block::iterator I = std::next(Op->getIterator());
    mlir::Block::iterator E = Op->getBlock()->end();

    // Skip over any invisible statement operations.
    while (I != E and not isVisibleStatement(&*I))
      ++I;

    // Prior to C23, labels cannot be placed at the end of a block:
    if (I == E)
      return true;

    // Prior to C23, labels cannot be placed preceding a declaration:
    if (mlir::isa<LocalVariableOp>(&*I))
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
      Tokens.emitKeyword(CTE::Keyword::BreakTo);
    else if (mlir::isa<ContinueToOp>(S))
      Tokens.emitKeyword(CTE::Keyword::ContinueTo);
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

  class CaseValueEmitter {
    CliftToCEmitter &Parent;
    mlir::Type Type;
    EnumType Enum;
    bool IsSigned;
    unsigned Radix;

  public:
    explicit CaseValueEmitter(CliftToCEmitter &Parent,
                              mlir::Type Type,
                              unsigned Radix) :
      CaseValueEmitter(Parent, Type, unwrapTypedefs(Type), Radix) {}

    void emit(uint64_t Value) const {
      if (Enum) {
        if (auto Enumerator = Enum.getFieldByValue(Value))
          return Parent.emitEnumImmediate(Value, Enum);

        Parent.emitCStyleCast(Type);
      }

      Parent.emitIntegerLiteral(Value, IsSigned, CStandardType::Int, Radix);
    }

  private:
    explicit CaseValueEmitter(CliftToCEmitter &Parent,
                              mlir::Type Type,
                              mlir::Type UnwrappedType,
                              unsigned Radix) :
      Parent(Parent),
      Type(Type),
      Enum(mlir::dyn_cast<EnumType>(UnwrappedType)),
      IsSigned(getUnderlyingIntegerType(UnwrappedType).isSigned()),
      Radix(Radix) {}
  };

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

      CaseValueEmitter CVE(*this, S.getConditionType(), Radix);
      for (unsigned I = 0, Count = S.getNumCases(); I < Count; ++I) {
        Tokens.emitKeyword(CTE::Keyword::Case);
        Tokens.emitSpace();

        CVE.emit(S.getCaseValue(I));

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

  RecursiveCoroutine<void> emitStatement(StatementOpInterface Statement) {
    mlir::Operation *Op = Statement.getOperation();

    if (auto Comments = getComments(Statement)) {
      FunctionOp ParentFunction = Op->getParentOfType<FunctionOp>();
      auto FLoc = pipeline::locationFromString(revng::ranks::Function,
                                               ParentFunction.getHandle());
      revng_assert(FLoc.has_value());

      for (auto [CommentIndex, CommentAttr] : llvm::enumerate(Comments)) {
        auto Location = pipeline::location(revng::ranks::StatementComment,
                                           FLoc->at(revng::ranks::Function),
                                           CommentIndex);

        auto Guard = Tokens.enterRegion(CTE::RegionKind::Commentable,
                                        Location.toString());

        // TODO: Add a comment formatting layer on top of CE.
        //       At least spaces at the start of each line would be nice.
        auto CE = Tokens.emitComment(CTE::CommentKind::Line);
        CE.emit(mlir::cast<mlir::StringAttr>(CommentAttr).getValue());
        CE.emit("\n");
      }
    }

    if (auto S = mlir::dyn_cast<LocalVariableOp>(Op))
      return emitLocalVariableDeclaration(S, /*Newline=*/true);

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
    for (mlir::Operation &Stmt : getVisibleStatementRange(R))
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

      rc_recur emitStatementRegion(Op.getBody());

      // TODO: emit a comment containing homeless variable names.
      //       See how old backend does it for reference.
    }

    Tokens.emitNewline();
  }
};

} // namespace

void decompile(FunctionOp F,
               ptml::CTokenEmitter &Emitter,
               CBackendConfiguration Configuration) {
  CliftToCEmitter(Emitter, getDataModel(F), Configuration).emitFunction(F);
}
