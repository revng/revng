//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"

#include "revng/CliftEmitC/CBackend.h"
#include "revng/CliftImportModel/ImportModel.h"
#include "revng/CliftTransforms/Passes.h"
#include "revng/Support/Debug.h"

namespace clift {
#define GEN_PASS_DEF_CLIFTEMITC
#include "revng/CliftTransforms/Passes.h.inc"
} // namespace clift

namespace {

struct EmitCPass : clift::impl::CliftEmitCBase<EmitCPass> {
  static std::unique_ptr<llvm::ToolOutputFile>
  tryOpenOutputFile(llvm::StringRef Filename) {
    std::string ErrorMessage;
    auto File = mlir::openOutputFile(Filename, &ErrorMessage);

    if (File)
      File->keep();
    else
      dbg << ErrorMessage << "\n";

    return File;
  }

  void runOnOperation() override {
    auto File = tryOpenOutputFile(Output);
    if (not File) {
      signalPassFailure();
      return;
    }

    const auto &Target = TargetCImplementation::Default;

    auto Tagging = static_cast<ptml::Tagging>(EmitTags.getValue());
    ptml::CTokenEmitter Emitter(File->os(), Tagging);

    getOperation()->walk([&Emitter](clift::FunctionOp Function) {
      if (not Function.isExternal())
        clift::decompile(Function, Emitter, Target);
    });
  }
};

} // namespace

clift::PassPtr<mlir::ModuleOp> clift::createEmitCPass() {
  return std::make_unique<EmitCPass>();
}
