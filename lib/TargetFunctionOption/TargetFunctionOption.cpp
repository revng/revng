/// \file TargetFunction.cpp
/// \brief File which defines an option later used by the `CDecompilerPass` and
///        by the `RestructureCFGPass`

//
// Copyright (c) rev.ng Srls 2017-2020.
//

// Local libraries includes
#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

using namespace llvm;

cl::opt<std::string> TargetFunction("single-decompilation",
                                    cl::desc("Function name to decompile"),
                                    cl::value_desc("function-name"));
