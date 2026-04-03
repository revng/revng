#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

namespace revng {

/// Performs initialization and shutdown steps for revng tools.
///
/// By default this performs the regular LLVM initialization steps.
/// This is required in order to initialize the stack trace printers on signal.
class InitRevng : public llvm::InitLLVM {
private:
  static inline bool Initialized = false;

public:
  InitRevng(int &Argc, char **&Argv, const char *Overview) :
    InitRevng(Argc, Argv, Overview, {}) {}

  InitRevng(int &Argc,
            char **&Argv,
            const char *Overview,
            llvm::ArrayRef<const llvm::cl::OptionCategory *> CategoriesToHide);

  ~InitRevng();

  InitRevng(const InitRevng &) = delete;
  InitRevng &operator=(const InitRevng &) = delete;
  InitRevng(InitRevng &&) = delete;
  InitRevng &operator=(InitRevng &&) = delete;

private:
  void initializeLLVMLibraries();
};

} // namespace revng
