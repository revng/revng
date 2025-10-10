#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/OpDefinition.h"

#include "revng/Clift/CliftOpInterfaces.h"

namespace mlir {
namespace OpTrait {
namespace clift {
namespace impl {

LogicalResult verifyNoFallthroughTrait(Operation *Op);

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
class ReturnsBoolean : public OpTrait::TraitBase<ConcreteType, ReturnsBoolean> {
  using Base = OpTrait::TraitBase<ConcreteType, ReturnsBoolean>;
};

} // namespace clift
} // namespace OpTrait
} // namespace mlir
