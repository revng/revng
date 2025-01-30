//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"

#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"
#include "revng/mlir/Dialect/Clift/Utils/CSemantics.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTVERIFYC
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

struct VerifyCPass : clift::impl::CliftVerifyCBase<VerifyCPass> {
  void runOnOperation() override {
    clift::TargetCImplementation Target = {
      .PointerSize = 8,
      .IntegerTypes = {
        { 1, clift::CIntegerKind::Char },
        { 2, clift::CIntegerKind::Short },
        { 4, clift::CIntegerKind::Int },
        { 8, clift::CIntegerKind::Long },
      },
    };

    if (mlir::failed(verifyCSemantics(getOperation(), Target)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<clift::ModuleOp>>
clift::createVerifyCPass() {
  return std::make_unique<VerifyCPass>();
}
