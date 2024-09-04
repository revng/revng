#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/OpDefinition.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftOpInterfaces.h"

namespace mlir {
namespace OpTrait {
namespace clift {
namespace impl {

LogicalResult verifyNoFallthroughTrait(Operation *Op);

} // namespace impl

template<typename UseType>
struct OneUseOfType {
  template<typename ConcreteType>
  class Impl : public OpTrait::TraitBase<ConcreteType, Impl> {
  private:
    using Base = OpTrait::TraitBase<ConcreteType, Impl>;

  public:
    static LogicalResult verifyTrait(Operation *Op) {
      static_assert(ConcreteType::template hasTrait<OneResult>(),
                    "expected operation to produce one result");

      mlir::Value Result = Op->getResult(0);

      auto OwnerLambda = [](mlir::OpOperand &Operand) {
        return mlir::isa<UseType>(Operand.getOwner());
      };
      const size_t NumUsesOfProvidedType = llvm::count_if(Result.getUses(),
                                                          OwnerLambda);

      if (NumUsesOfProvidedType > 1)
        return Op->emitOpError() << "expects to have a single use which is a "
                                 << UseType::getOperationName();
      return success();
    }
  };
};

template<typename ConcreteType>
class NoFallthrough : public OpTrait::TraitBase<ConcreteType, NoFallthrough> {
  using Base = OpTrait::TraitBase<ConcreteType, NoFallthrough>;

public:
  static LogicalResult verifyTrait(Operation *const Op) {
    return impl::verifyNoFallthroughTrait(Op);
  }
};

} // namespace clift
} // namespace OpTrait
} // namespace mlir
