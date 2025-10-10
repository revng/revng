#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/Clift.h"

namespace mlir::clift {

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

inline BlockPosition getJumpTarget(JumpStatementOpInterface Jump) {
  mlir::Operation *Op = Jump.getLabelAssignmentOp();
  return BlockPosition::get(Op);
}

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

inline YieldOp getYieldOp(mlir::Region &R) {
  return getLastOp<YieldOp>(R);
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
    return Op->template hasTrait<mlir::OpTrait::clift::NoFallthrough>();
  });
}

} // namespace mlir::clift
