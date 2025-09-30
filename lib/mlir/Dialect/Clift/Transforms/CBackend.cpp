//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"

#include "revng/Support/Debug.h"
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h"
#include "revng/mlir/Dialect/Clift/Utils/CBackend.h"
#include "revng/mlir/Dialect/Clift/Utils/ImportModel.h"

namespace mlir {
namespace clift {
#define GEN_PASS_DEF_CLIFTEMITC
#include "revng/mlir/Dialect/Clift/Transforms/Passes.h.inc"
} // namespace clift
} // namespace mlir

namespace clift = mlir::clift;

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

    CTokenEmitter Emitter(File->os(),
                          static_cast<ptml::Tagging>(EmitTags.getValue()));

    getOperation()->walk([&Emitter](clift::FunctionOp Function) {
      if (not Function.isExternal())
        clift::decompile(Function, Emitter, Target);
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> clift::createEmitCPass() {
  return std::make_unique<EmitCPass>();
}
