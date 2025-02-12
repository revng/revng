#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"

namespace revng::options {

// Type inlining is enabled by default.
extern llvm::cl::opt<bool> DisableTypeInlining;

// Stack frame inlining is enabled by default.
extern llvm::cl::opt<bool> DisableStackFrameInlining;

} // namespace revng::options
