#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/InitLLVM.h"

namespace revng {

/// \brief Performs initialization and shutdown steps for revng tools.
///
/// By default this performs the regular LLVM initialization steps.
/// This is required in order to initialize the stack trace printers on signal.
class InitRevng : public llvm::InitLLVM {
public:
  InitRevng(int &Argc,
            const char **&Argv,
            bool InstallPipeSignalExitHandler = true) :
    InitLLVM(Argc, Argv, InstallPipeSignalExitHandler) {}

  InitRevng(int &Argc, char **&Argv, bool InstallPipeSignalExitHandler = true) :
    InitLLVM(Argc, Argv, InstallPipeSignalExitHandler) {}
};

} // namespace revng
