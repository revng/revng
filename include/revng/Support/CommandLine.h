#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iosfwd>

#include "llvm/Support/CommandLine.h"

extern llvm::cl::OptionCategory MainCategory;

/// This is the path of the input binary. CLI tools should populate this on
/// every invocations. Users of this variable need to account for it being
/// possibly empty.
extern std::string InputPath;

std::ostream &pathToStream(const std::string &Path, std::ofstream &File);
