#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"

#include "revng/Support/Statistics.h"

namespace revng {

/// Performs initialization and shutdown steps for revng tools.
///
/// By default this performs the regular LLVM initialization steps.
/// This is required in order to initialize the stack trace printers on signal.
class InitRevng : public llvm::InitLLVM {
public:
  InitRevng(int &Argc,
            auto **&Argv,
            const char *Overview,
            llvm::ArrayRef<const llvm::cl::OptionCategory *> CategoriesToHide) :
    InitLLVM(Argc, Argv, true) {

    OnQuit->install();

    llvm::setBugReportMsg("PLEASE submit a bug report to "
                          "https://github.com/revng/revng and include the "
                          "crash backtrace\n");

    if (not CategoriesToHide.empty()) {
      // NOLINTNEXTLINE
      llvm::cl::HideUnrelatedOptions(CategoriesToHide);
    }

    // NOLINTNEXTLINE
    bool Result = llvm::cl::ParseCommandLineOptions(Argc,
                                                    Argv,
                                                    Overview,
                                                    nullptr,
                                                    "REVNG_OPTIONS");

    if (not Result)
      std::exit(EXIT_FAILURE);
  }

  ~InitRevng() { OnQuit->quit(); }
};

} // namespace revng
