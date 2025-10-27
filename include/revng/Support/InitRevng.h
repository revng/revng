#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugInfoPreservation.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"

#include "revng/Support/Statistics.h"

namespace revng {

/// Performs initialization and shutdown steps for revng tools.
///
/// By default this performs the regular LLVM initialization steps.
/// This is required in order to initialize the stack trace printers on signal.
class InitRevng : public llvm::InitLLVM {
private:
  static inline bool Initialized = false;

public:
  InitRevng(int &Argc,
            auto **&Argv,
            const char *Overview,
            llvm::ArrayRef<const llvm::cl::OptionCategory *> CategoriesToHide) :
    InitLLVM(Argc, Argv, true) {

    revng_assert(not Initialized);
    Initialized = true;

    OnQuit->install();
    initializeLLVMLibraries();

    llvm::setBugReportMsg("PLEASE submit a bug report to "
                          "https://github.com/revng/revng and include the "
                          "crash backtrace\n");

    if (not CategoriesToHide.empty()) {
      // NOLINTNEXTLINE
      llvm::cl::HideUnrelatedOptions(CategoriesToHide);
    }

    // The EnvVar option in llvm::cl::ParseCommandLineOptions prepends the
    // contents of the EnvVar environment variable to the argv, this is
    // undesirable since we have `-load`s in the argv. Here we parse the env
    // manually (if present) and append the command-line options at the end of
    // the argv.
    llvm::BumpPtrAllocator A;
    llvm::StringSaver Saver(A);
    llvm::SmallVector<const char *, 0> Arguments(Argv, Argv + Argc);
    if (auto EnvValue = llvm::sys::Process::GetEnv("REVNG_OPTIONS")) {
      llvm::cl::TokenizeGNUCommandLine(*EnvValue, Saver, Arguments);
    }

    // NOLINTNEXTLINE
    bool Result = llvm::cl::ParseCommandLineOptions(Arguments.size(),
                                                    Arguments.data(),
                                                    Overview);

    if (not Result)
      std::exit(EXIT_FAILURE);

    // Force-enable `--enable-strict-debug-information-preservation-style` for
    // revng binaries even if it wasn't specified.
    llvm::EnableStrictDebugInformationPreservationStyle.setInitialValue(true);
  }

  ~InitRevng() { OnQuit->quit(); }

private:
  void initializeLLVMLibraries();
};

} // namespace revng
