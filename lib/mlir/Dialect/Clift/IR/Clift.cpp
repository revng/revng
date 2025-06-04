//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "revng/mlir/Dialect/Clift/IR/Clift.h"
#include "revng/mlir/Dialect/Clift/IR/CliftTypes.h"

// comment to force generated files to be included be after regular includes
#include "revng/mlir/Dialect/Clift/IR/CliftOpsDialect.cpp.inc"

namespace {

class CliftOpAsmDialectInterface final : public mlir::OpAsmDialectInterface {
public:
  using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Attribute Attr, llvm::raw_ostream &OS) const {
    if (auto Aliasable = mlir::dyn_cast<mlir::clift::AliasableAttr>(Attr)) {
      if (Aliasable.getAlias(OS))
        return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }

  AliasResult getAlias(mlir::Type Type, llvm::raw_ostream &OS) const {
    if (auto Aliasable = mlir::dyn_cast<mlir::clift::AliasableType>(Type)) {
      if (Aliasable.getAlias(OS))
        return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }
};

} // namespace

void mlir::clift::CliftDialect::initialize() {
  registerTypes();
  registerOperations();
  registerAttributes();
  addInterfaces<CliftOpAsmDialectInterface>();
}
