#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iosfwd>

#include "llvm/Support/CommandLine.h"

extern llvm::cl::OptionCategory MainCategory;

// Popular option
extern llvm::cl::opt<bool> IgnoreDebugSymbols;

std::ostream &pathToStream(const std::string &Path, std::ofstream &File);
