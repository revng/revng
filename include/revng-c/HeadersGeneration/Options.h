#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"

namespace revng::options {

// Disabled by default.
extern llvm::cl::opt<bool> EnableTypeInlining;

// Enabled by default.
extern llvm::cl::opt<bool> DisableStackFrameInlining;

} // namespace revng::options
