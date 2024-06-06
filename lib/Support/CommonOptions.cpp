/// \file CommonOptions.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/CommonOptions.h"

using namespace llvm;

cl::opt<bool> DebugNames("debug-names",
                         cl::desc("Use friendly names in non-user artifacts"),
                         cl::init(false));
