#pragma once

#include "revng/mlir/Dialect/Clift/IR/CliftOps.h"

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

  static BlockPosition getEnd(mlir::Region &R) {
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

  explicit operator bool() const {
    return Block != nullptr;
  }

  friend bool operator==(BlockPosition const &,
                         BlockPosition const &) = default;
};

inline BlockPosition getGotoTarget(clift::GoToOp Goto) {
  return BlockPosition::get(Goto.getAssignLabelOp().getOperation());
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

template<typename Operation = mlir::Operation *>
Operation getOnlyOperation(mlir::Region &R) {
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

template<typename OpT = mlir::Operation *, typename PredicateT>
OpT getLeadingOp(mlir::Region &Region, PredicateT &&Predicate) {
  if (Region.empty())
    return {};

  revng_assert(Region.hasOneBlock());
  mlir::Block &Block = Region.front();

  if (Block.empty())
    return {};

  mlir::Operation *Op = &Block.front();
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
OpT getLeadingOp(mlir::Region &Region) {
  return getLeadingOp<OpT>(Region, [](OpT) { return true; });
}

template<typename OpT = mlir::Operation *, typename PredicateT>
OpT getTrailingOp(mlir::Region &Region, PredicateT &&Predicate) {
  if (Region.empty())
    return {};

  revng_assert(Region.hasOneBlock());
  mlir::Block &Block = Region.front();

  if (Block.empty())
    return {};

  mlir::Operation *Op = &Block.back();
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
OpT getTrailingOp(mlir::Region &Region) {
  return getTrailingOp<OpT>(Region, [](OpT) { return true; });
}


inline YieldOp getYieldOp(mlir::Region &R) {
  return getTrailingOp<YieldOp>(R);
}

template<typename ... ArgsT>
StatementOpInterface getTrailingStatement(mlir::Region &R, ArgsT &&...Args) {
  return getTrailingOp<StatementOpInterface>(R, std::forward<ArgsT>(Args)...);
}

inline StatementOpInterface getTrailingJumpOp(mlir::Region &R) {
  return getTrailingOp<StatementOpInterface>(R, [](auto Op) {
    return Op->template hasTrait<mlir::OpTrait::clift::NoFallthrough>();
  });
}

} // namespace mlir::clift
