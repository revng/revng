//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "revng-c/mlir/Dialect/Clift/IR/CliftOps.h"
#include "revng-c/mlir/Dialect/Clift/Transforms/Passes.h"
#include "revng-c/mlir/Dialect/Clift/Utils/ImportModel.h"
#include "revng-c/mlir/Dialect/Clift/Utils/ImportReachableModelTypes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTIMPORTMODELTYPES
#include "revng-c/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

static void importAllModelTypes(mlir::ModuleOp Module,
                                const model::Binary &Model) {
  mlir::MLIRContext *const Context = Module->getContext();

  const auto EmitError = [&]() -> mlir::InFlightDiagnostic {
    return Context->getDiagEngine().emit(mlir::UnknownLoc::get(Context),
                                         mlir::DiagnosticSeverity::Error);
  };

  mlir::OpBuilder Builder(Module.getRegion());
  for (const auto &ModelType : Model.Types()) {
    auto CliftType = clift::importModelType(EmitError, *Context, *ModelType);
    Builder.create<clift::UndefOp>(mlir::UnknownLoc::get(Context), CliftType);
  }
}

using clift::impl::CliftImportModelTypesBase;
struct ImportModelTypesPass : CliftImportModelTypesBase<ImportModelTypesPass> {
  void runOnOperation() override {
    if (ImportAll)
      importAllModelTypes(getOperation(), *Model);
    else
      clift::importReachableModelTypes(getOperation(), *Model);
  };
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
clift::createImportModelTypesPass() {
  return std::make_unique<ImportModelTypesPass>();
}
