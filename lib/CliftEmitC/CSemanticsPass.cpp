//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "mlir/Pass/Pass.h"

#include "revng/CliftEmitC/CSemantics.h"
#include "revng/CliftTransforms/Passes.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTVERIFYC
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

namespace {

struct VerifyCPass : clift::impl::CliftVerifyCBase<VerifyCPass> {
  void runOnOperation() override {
    mlir::ModuleOp Module = getOperation();

    if (mlir::failed(verifyCSemantics(Module)))
      signalPassFailure();
  }
};

} // namespace

clift::PassPtr<mlir::ModuleOp> clift::createVerifyCPass() {
  return std::make_unique<VerifyCPass>();
}
