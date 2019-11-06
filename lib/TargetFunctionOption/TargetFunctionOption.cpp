/// \file TargetFunction.cpp
/// \brief File which defines an option later used by the `CDecompilerPass` and
///        by the `RestructureCFGPass`

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Local libraries includes
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

using namespace llvm;

cl::opt<std::string> TargetFunction("single-decompilation",
                                    cl::desc("Function name to decompile"),
                                    cl::value_desc("function-name"));
