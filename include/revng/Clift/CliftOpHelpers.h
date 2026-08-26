#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"

namespace clift {

struct BlockPosition {
  mlir::Block *Block;
  mlir::Block::iterator Pos;

  static BlockPosition get(mlir::Operation *Op) {
    return BlockPosition{ Op->getBlock(), Op->getIterator() };
  }

  static BlockPosition getNext(mlir::Operation *Op) {
    return BlockPosition{ Op->getBlock(), std::next(Op->getIterator()) };
  }

  static BlockPosition getBegin(mlir::Region &R) {
    revng_assert(not R.empty());
    return { &R.front(), R.front().begin() };
  }

  static BlockPosition getEnd(mlir::Region &R) {
    revng_assert(R.hasOneBlock());
    return { &R.front(), R.front().end() };
  }

  template<typename OpT = mlir::Operation *>
  OpT getOperation() const {
    if (Block == nullptr)
      return {};
    if (Pos == Block->end())
      return {};
    return mlir::dyn_cast<OpT>(&*Pos);
  }

  explicit operator bool() const { return Block != nullptr; }

  friend bool operator==(BlockPosition const &,
                         BlockPosition const &) = default;
};

inline bool isEmptyRegionOrBlock(mlir::Region &R) {
  return R.empty() or R.front().empty();
}

inline bool hasEmptyBlock(mlir::Region &R) {
  return not R.empty() and R.front().empty();
}

inline bool isFirstInBlock(mlir::Operation *Op) {
  return Op->getIterator() == Op->getBlock()->begin();
}

inline bool isLastInBlock(mlir::Operation *Op) {
  return std::next(Op->getIterator()) == Op->getBlock()->end();
}

inline mlir::Block *getOnlyBlock(mlir::Region &R) {
  return R.hasOneBlock() ? &R.front() : nullptr;
}

inline mlir::Block *extractOnlyBlock(mlir::Region &R) {
  mlir::Block *Block = getOnlyBlock(R);
  if (Block != nullptr)
    R.getBlocks().remove(Block);
  return Block;
}

inline void setOnlyBlock(mlir::Region &R, mlir::Block *Block) {
  if (not R.empty())
    R.getBlocks().clear();
  if (Block != nullptr)
    R.push_back(Block);
}

template<typename OpT = mlir::Operation *, typename PredicateT>
OpT getOnlyOpIf(mlir::Region &R, PredicateT &&Predicate) {
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

  if constexpr (std::is_same_v<OpT, mlir::Operation *>) {
    if (Predicate(Op))
      return Op;
  } else {
    if (auto Op2 = mlir::dyn_cast<OpT>(Op)) {
      if (Predicate(Op2))
        return Op2;
    }
  }

  return {};
}

template<typename OpT = mlir::Operation *>
OpT getOnlyOp(mlir::Region &R) {
  return getOnlyOpIf<OpT>(R, [](OpT) { return true; });
}

template<typename OpT = mlir::Operation *, typename PredicateT>
OpT getFirstOpIf(mlir::Region &R, PredicateT &&Predicate) {
  if (R.empty())
    return {};

  revng_assert(R.hasOneBlock());
  mlir::Block &B = R.front();

  if (B.empty())
    return {};

  mlir::Operation *Op = &B.front();
  if constexpr (std::is_same_v<OpT, mlir::Operation *>) {
    if (Predicate(Op))
      return Op;
  } else {
    if (auto Op2 = mlir::dyn_cast<OpT>(Op)) {
      if (Predicate(Op2))
        return Op2;
    }
  }

  return {};
}

template<typename OpT = mlir::Operation *>
OpT getFirstOp(mlir::Region &Region) {
  return getFirstOpIf<OpT>(Region, [](OpT) { return true; });
}

template<typename OpT = mlir::Operation *, typename PredicateT>
OpT getLastOpIf(mlir::Region &R, PredicateT &&Predicate) {
  if (R.empty())
    return {};

  revng_assert(R.hasOneBlock());
  mlir::Block &B = R.front();

  if (B.empty())
    return {};

  mlir::Operation *Op = &B.back();
  if constexpr (std::is_same_v<OpT, mlir::Operation *>) {
    if (Predicate(Op))
      return Op;
  } else {
    if (auto Op2 = mlir::dyn_cast<OpT>(Op)) {
      if (Predicate(Op2))
        return Op2;
    }
  }

  return {};
}

template<typename OpT = mlir::Operation *>
OpT getLastOp(mlir::Region &Region) {
  return getLastOpIf<OpT>(Region, [](OpT) { return true; });
}

template<typename OpT = mlir::Operation *, typename PredicateT>
OpT getNextOpIf(mlir::Operation *Op, PredicateT &&Predicate) {
  auto NextIterator = std::next(Op->getIterator());
  if (NextIterator == Op->getBlock()->end())
    return {};

  mlir::Operation *NextOp = &*NextIterator;
  if constexpr (std::is_same_v<OpT, mlir::Operation *>) {
    if (Predicate(NextOp))
      return NextOp;
  } else {
    if (auto NextOp2 = mlir::dyn_cast<OpT>(NextOp)) {
      if (Predicate(NextOp2))
        return NextOp2;
    }
  }

  return {};
}

template<typename OpT = mlir::Operation *>
OpT getNextOp(mlir::Operation *Op) {
  return getNextOpIf<OpT>(Op, [](OpT) { return true; });
}

//===----------------------------- Statements -----------------------------===//

inline BlockPosition getJumpTarget(JumpStatementOpInterface Jump) {
  mlir::Operation *Op = Jump.getLabelAssignmentOp();

  if (auto Loop = mlir::dyn_cast<LoopOpInterface>(Op)) {
    auto Label = Jump.getLabel();
    if (Label == Loop.getBreakLabel())
      return BlockPosition::getNext(Loop);
    if (Label == Loop.getContinueLabel())
      return BlockPosition::getEnd(Loop.getBody());
  }

  return BlockPosition::get(Op);
}

/// Returns the innermost loop enclosing \p Op, or a null interface if there is
/// none within the current function. If \p CrossedSwitch is non-null, it is set
/// to true whenever a switch statement is nested strictly between \p Op and the
/// returned loop. This mirrors the reachability of a plain C break statement,
/// which is captured by the innermost enclosing loop *or* switch: a break can
/// only stand in for a break_to targeting a loop when no switch is crossed.
inline LoopOpInterface getEnclosingLoop(mlir::Operation *Op,
                                        bool *CrossedSwitch = nullptr) {
  if (CrossedSwitch != nullptr)
    *CrossedSwitch = false;

  for (mlir::Operation *Parent = Op->getParentOp(); Parent != nullptr;
       Parent = Parent->getParentOp()) {
    if (auto Loop = mlir::dyn_cast<LoopOpInterface>(Parent))
      return Loop;

    if (CrossedSwitch != nullptr and mlir::isa<SwitchOp>(Parent))
      *CrossedSwitch = true;
  }

  return {};
}

template<typename PredicateT>
StatementOpInterface
getLastStatementIf(mlir::Region &R, PredicateT &&Predicate) {
  return getLastOpIf<StatementOpInterface>(R,
                                           std::forward<PredicateT>(Predicate));
}

inline StatementOpInterface getLastStatement(mlir::Region &R) {
  return getLastOp<StatementOpInterface>(R);
}

inline StatementOpInterface getLastNoFallthroughStatement(mlir::Region &R) {
  return getLastStatementIf(R, [](auto Op) {
    return Op->template hasTrait<clift::NoFallthrough>();
  });
}

inline NoFallthroughKind isIndirectlyNoFallthrough(mlir::Region &R) {
  StatementOpInterface Op = getLastStatement(R);
  if (not Op)
    return NoFallthroughKind::FallsThrough;

  // A statement carrying the NoFallthrough trait is directly non-fallthrough;
  // its concrete kind classifies the region.
  if (Op->template hasTrait<clift::NoFallthrough>()) {
    if (mlir::isa<ContinueToOp>(Op))
      return NoFallthroughKind::Continue;
    if (mlir::isa<BreakToOp>(Op))
      return NoFallthroughKind::Break;
    if (mlir::isa<GotoOp>(Op))
      return NoFallthroughKind::Goto;
    revng_assert(mlir::isa<ReturnOp>(Op));
    return NoFallthroughKind::Return;
  }

  // Otherwise the region can be non-fallthrough only indirectly, through a
  // nested branch or block; defer to the operation's own classification.
  return Op.isIndirectlyNoFallthrough();
}

// A region indirectly falls through when control can reach its end, whether
// directly or through the statement it ends in. A block-less region - a missing
// else or default, or an empty `{}` case body - also falls through.
inline bool indirectlyFallsThrough(mlir::Region &R) {
  return isIndirectlyNoFallthrough(R) == NoFallthroughKind::FallsThrough;
}

//===----------------------------- Expressions ----------------------------===//

inline YieldOp getYieldOp(mlir::Region &R) {
  return getLastOp<YieldOp>(R);
}

inline ExpressionOpInterface getRootExpression(mlir::Region &R) {
  if (auto Yield = getYieldOp(R))
    return Yield.getValue().getDefiningOp<ExpressionOpInterface>();
  return {};
}

inline bool isBooleanExpression(mlir::Value Value) {
  mlir::Operation *Op = Value.getDefiningOp();
  return Op and Op->hasTrait<clift::ReturnsBoolean>();
}

inline mlir::OpOperand *getOnlyUse(mlir::Value Value) {
  auto Begin = Value.use_begin();
  auto End = Value.use_end();

  if (Begin == End)
    return nullptr;

  return &*Begin;
}

template<typename OpT = mlir::Operation *>
OpT getOnlyUser(mlir::Value Value) {
  if (mlir::OpOperand *Operand = getOnlyUse(Value)) {
    if constexpr (std::is_same_v<OpT, mlir::Operation *>) {
      return Operand->getOwner();
    } else {
      return mlir::dyn_cast<OpT>(Operand->getOwner());
    }
  }
  return nullptr;
}

//===-------------------------- Expression usage --------------------------===//

/// Returns true if the value is discarded. A value might be discarded by for
/// instance by an expression statement or a comma expression.
bool isDiscarded(mlir::Value Value);

/// Returns true if the value is boolean-tested. A value might be boolean-tested
/// for instance by a control flow condition, a ternary expression or a logical
/// expression.
bool isBooleanTested(mlir::Value Value);

} // namespace clift
