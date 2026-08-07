#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/IRHelper.h"

/// The arguments of `function_call`
enum class FunctionCallArgument {
  Callee,
  Fallthrough,
  FallthroughAddress,
  LinkRegister
};

/// Marks the terminator of a basic block performing a function call
inline IRHelper<FunctionCallArgument> FunctionCallMarker("function_call");
