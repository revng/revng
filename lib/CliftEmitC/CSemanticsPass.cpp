//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"

#include "revng/CliftEmitC/CSemantics.h"
#include "revng/CliftTransforms/Passes.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTVERIFYC
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

namespace {

struct VerifyCPass : clift::impl::CliftVerifyCBase<VerifyCPass> {
  void runOnOperation() override {
    const auto &Target = TargetCImplementation::Default;

    if (mlir::failed(clift::verifyCSemantics(getOperation(), Target)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
clift::createVerifyCPass() {
  return std::make_unique<VerifyCPass>();
}
