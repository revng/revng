/// \file TargetFunction.cpp
/// \brief File which defines an option later used by the `CDecompilerPass` and
///        by the `RestructureCFGPass`

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//

#include "revng-c/TargetFunctionOption/TargetFunctionOption.h"

using namespace llvm;

cl::opt<std::string> TargetFunction("single-decompilation",
                                    cl::desc("Name of the function to "
                                             "decompile"),
                                    cl::value_desc("function name"),
                                    cl::NumOccurrencesFlag::Optional);
