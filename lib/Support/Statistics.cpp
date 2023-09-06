/// \file Statistics.cpp
/// \brief Implementation of the statistics collection framework.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/CommandLine.h"
#include "revng/Support/Statistics.h"

namespace cl = llvm::cl;

cl::opt<bool> Statistics("statistics",
                         cl::desc("print statistics upon exit or "
                                  "SIGINT. Use "
                                  "this argument, ignore -stats."),
                         cl::cat(MainCategory));
