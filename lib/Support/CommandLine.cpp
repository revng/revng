/// \file CommandLine.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <iostream>

#include "revng/Support/CommandLine.h"

namespace cl = llvm::cl;

cl::OptionCategory MainCategory("Options", "");
std::string InputPath;
