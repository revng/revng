//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "mlir/IR/OpImplementation.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpsDialect.cpp.inc"
#include "revng-c/mlir/Dialect/Clift/IR/CliftTypes.h"

class TypeAliasASMInterface : public mlir::OpAsmDialectInterface {
public:
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Type Type, llvm::raw_ostream &OS) const final {

    return AliasResult::NoAlias;
  }
};

void mlir::clift::CliftDialect::initialize() {
  registerTypes();
  registerOperations();
  addInterfaces<TypeAliasASMInterface>();
}
