//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/SourceMgr.h"

#include "revng/Support/CommandLine.h"
#include "revng/Support/Error.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/Tar.h"

static llvm::cl::list<std::string> Arguments(llvm::cl::Positional,
                                             llvm::cl::ZeroOrMore,
                                             llvm::cl::desc("FILES"),
                                             llvm::cl::cat(MainCategory));

static llvm::ExitOnError AbortOnError;

int main(int argc, char *argv[]) {
  revng::InitRevng X(argc, argv, "", { &MainCategory });

  llvm::LLVMContext Context;
  std::unique_ptr<llvm::Module> OutputModule;

  for (llvm::StringRef Filename : Arguments) {
    llvm::SMDiagnostic Err;
    auto Module = llvm::parseIRFile(Filename, Err, Context);
    revng_assert(Module != nullptr);

    if (OutputModule != nullptr) {
      // Make the existing module be linked to the new module, this allows
      // preserving the order of definitions as the order of the supplied files
      std::swap(Module, OutputModule);
      linkFunctionModules(std::move(Module), OutputModule);
    } else {
      OutputModule = std::move(Module);
    }
  }

  revng_assert(OutputModule != nullptr);
  revng::verify(OutputModule.get());
  {
    std::error_code EC;
    llvm::raw_fd_ostream OS("-", EC);
    revng_assert(not EC);

    OutputModule->print(OS, nullptr);
  }

  return EXIT_SUCCESS;
}
