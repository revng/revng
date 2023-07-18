//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "revng-c/mlir/Dialect/Clift/IR/Clift.h"
#include "revng-c/mlir/Dialect/Clift/IR/CliftTypes.h"

// comment to force generated files to be included be after regular includes
#include "revng-c/mlir/Dialect/Clift/IR/CliftOpsDialect.cpp.inc"

class TypeAliasASMInterface : public mlir::OpAsmDialectInterface {
public:
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Attribute Attr,
                       llvm::raw_ostream &OS) const final {
    if (auto Casted = Attr.dyn_cast<mlir::clift::AliasableAttr>()) {
      if (not Casted.getAlias().empty()) {
        OS << Casted.getAlias();
        return AliasResult::FinalAlias;
      }
    }

    return AliasResult::NoAlias;
  }

  AliasResult getAlias(mlir::Type Type, llvm::raw_ostream &OS) const final {
    if (auto Casted = Type.dyn_cast<mlir::clift::AliasableType>()) {
      if (not Casted.getAlias().empty()) {
        OS << Casted.getAlias();
        return AliasResult::FinalAlias;
      }
    }

    return AliasResult::NoAlias;
  }
};

void mlir::clift::CliftDialect::initialize() {
  registerTypes();
  registerOperations();
  registerAttributes();
  addInterfaces<TypeAliasASMInterface>();
}
