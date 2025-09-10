//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <mutex>

#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"

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

// This forces all accesses to the output file to be synchronised.
template<typename T>
class OutputFileWrapper {
public:
  OutputFileWrapper() = default;
  OutputFileWrapper(const OutputFileWrapper &) = delete;
  OutputFileWrapper &operator=(const OutputFileWrapper &) = delete;

  template<typename CallableType>
  decltype(auto) use(CallableType &&Callable) {
    std::lock_guard Lock(Mutex);
    return static_cast<CallableType &&>(Callable)(Value);
  }

private:
  std::mutex Mutex;
  T Value;
};

struct EmitCPass : clift::impl::CliftEmitCBase<EmitCPass> {
  using OutputFilePtr = std::unique_ptr<llvm::ToolOutputFile>;
  std::shared_ptr<OutputFileWrapper<OutputFilePtr>> OutputFile;

  EmitCPass() :
    OutputFile(std::make_shared<OutputFileWrapper<OutputFilePtr>>()) {}

  bool tryOpenOutputFile() {
    return OutputFile->use([&](auto &File) -> bool {
      if (File == nullptr) {
        std::string ErrorMessage;
        File = mlir::openOutputFile(Output, &ErrorMessage);

        if (File != nullptr) {
          File->keep();
        } else {
          llvm::errs() << ErrorMessage << "\n";
          signalPassFailure();
        }
      }
      return File != nullptr;
    });
  }

  void writeToOutputFile(llvm::StringRef Content) {
    OutputFile->use([&](const auto &File) {
      revng_assert(File != nullptr);
      File->os() << Content << '\n';
    });
  }

  void runOnOperation() override {
    const auto &Target = TargetCImplementation::Default;

    if (not tryOpenOutputFile())
      return;

    std::string DecompiledString;
    {
      llvm::raw_string_ostream DecompileOS(DecompiledString);
      CEmitter Emitter(DecompileOS,
                       static_cast<ptml::Tagging>(EmitTags.getValue()));

      getOperation()->walk([&](clift::FunctionOp Function) {
        if (not Function.isExternal())
          clift::decompile(Function, Emitter, Target);
      });
    }

    writeToOutputFile(DecompiledString);
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> clift::createEmitCPass() {
  return std::make_unique<EmitCPass>();
}
