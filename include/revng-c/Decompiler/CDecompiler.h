#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Module;
} // end namespace llvm

std::string
decompileFunction(const llvm::Module *M, const llvm::StringRef FunctionName);
