#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <iosfwd>

#include "llvm/Support/CommandLine.h"

extern llvm::cl::OptionCategory MainCategory;

std::ostream &pathToStream(const std::string &Path, std::ofstream &File);
