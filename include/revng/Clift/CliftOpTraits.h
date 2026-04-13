#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/OpDefinition.h"

namespace clift {
namespace impl {

mlir::LogicalResult verifyNoFallthroughTrait(mlir::Operation *Op);
mlir::LogicalResult verifyAssignsLoopLabelsTrait(mlir::Operation *Op);

} // namespace impl

template<typename ConcreteType>
class NoFallthrough
  : public mlir::OpTrait::TraitBase<ConcreteType, NoFallthrough> {
  using Base = mlir::OpTrait::TraitBase<ConcreteType, NoFallthrough>;

public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *const Op) {
    return impl::verifyNoFallthroughTrait(Op);
  }
};

template<typename ConcreteType>
class AssignsLoopLabels
  : public mlir::OpTrait::TraitBase<ConcreteType, AssignsLoopLabels> {
  using Base = mlir::OpTrait::TraitBase<ConcreteType, AssignsLoopLabels>;

public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *const Op) {
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
class ReturnsBoolean
  : public mlir::OpTrait::TraitBase<ConcreteType, ReturnsBoolean> {
  using Base = mlir::OpTrait::TraitBase<ConcreteType, ReturnsBoolean>;
};

} // namespace clift
