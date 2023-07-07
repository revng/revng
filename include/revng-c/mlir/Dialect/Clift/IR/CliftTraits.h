#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace OpTrait {

template<typename UseType>
struct OneUseOfType {
  template<typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, Impl> {
  private:
    using Base = TraitBase<ConcreteType, Impl>;

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

} // namespace OpTrait
} // namespace mlir
