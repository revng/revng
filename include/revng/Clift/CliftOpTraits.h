#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {
namespace clift {
namespace impl {

LogicalResult verifyNoFallthroughTrait(Operation *Op);
LogicalResult verifyAssignsLoopLabelsTrait(Operation *Op);

} // namespace impl

template<typename ConcreteType>
class NoFallthrough : public OpTrait::TraitBase<ConcreteType, NoFallthrough> {
  using Base = OpTrait::TraitBase<ConcreteType, NoFallthrough>;

public:
  static LogicalResult verifyTrait(Operation *const Op) {
    return impl::verifyNoFallthroughTrait(Op);
  }
};

template<typename ConcreteType>
class AssignsLoopLabels
  : public OpTrait::TraitBase<ConcreteType, AssignsLoopLabels> {
  using Base = OpTrait::TraitBase<ConcreteType, AssignsLoopLabels>;

public:
  static LogicalResult verifyTrait(Operation *const Op) {
    return impl::verifyAssignsLoopLabelsTrait(Op);
  }

  unsigned getAssignedLabelCount() {
    auto Op = mlir::cast<ConcreteType>(this->getOperation());
    return std::popcount(Op.getLabelMask());
  }

  mlir::Value getAssignedLabel(unsigned Index) {
    return this->getOperation()->getOperand(Index);
  }
};

template<typename ConcreteType>
class ReturnsBoolean : public OpTrait::TraitBase<ConcreteType, ReturnsBoolean> {
  using Base = OpTrait::TraitBase<ConcreteType, ReturnsBoolean>;
};

} // namespace clift
} // namespace OpTrait
} // namespace mlir
