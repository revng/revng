#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/LLVMContainer.h"

namespace revng::pypeline {

/// Utility piperun argument for LLVM pass-like pipes that take a single
/// LLVMFunctionContainer and also need the model to run
using SingleLLVMFunctionsArgument = TypeList<
  PipeRunArgument<LLVMFunctionContainer, "Module", "function LLVM module(s)">>;

} // namespace revng::pypeline
