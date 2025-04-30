#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/Generated/Early/FunctionEdgeType.h"

// TODO: we need to handle noreturn function calls

namespace efa::FunctionEdgeType {

inline bool isCall(Values V) {
  switch (V) {
  case Count:
    revng_abort();
    break;

  case FunctionCall:
    return true;

  case Invalid:
  case DirectBranch:
  case Return:
  case BrokenReturn:
  case LongJmp:
  case Killer:
  case Unreachable:
    return false;
  }
}

} // namespace efa::FunctionEdgeType

#include "revng/EarlyFunctionAnalysis/Generated/Late/FunctionEdgeType.h"
