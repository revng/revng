/// \file debug.cpp
/// \brief Implementation of the debug framework

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng/Support/CommandLine.h"

namespace cl = llvm::cl;

cl::OptionCategory MainCategory("Options", "");

cl::opt<bool> UseDebugSymbols("use-debug-symbols",
                              cl::desc("use section and symbol function "
                                       "informations, if available"),
                              cl::cat(MainCategory));
static cl::alias A1("S",
                    cl::desc("Alias for -use-debug-symbols"),
                    cl::aliasopt(UseDebugSymbols),
                    cl::cat(MainCategory));
