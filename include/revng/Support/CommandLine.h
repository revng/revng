#ifndef COMMANDLINE_H
#define COMMANDLINE_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <iosfwd>

// LLVM includes
#include "llvm/Support/CommandLine.h"

extern llvm::cl::OptionCategory MainCategory;

// Popular option
extern llvm::cl::opt<bool> UseDebugSymbols;

std::ostream &pathToStream(const std::string &Path, std::ofstream &File);

#endif // COMMANDLINE_H
