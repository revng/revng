#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/FunctionEdgeType.h"

#include "revng/Yield/Generated/Early/FunctionEdgeType.h"

// TODO: we need to handle noreturn function calls

namespace yield::FunctionEdgeType {

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
  case Unexpected:
  case Unreachable:
    return false;
  }
}

inline Values from(efa::FunctionEdgeType::Values V) {
  switch (V) {
  case efa::FunctionEdgeType::Count:
    return Count;
  case efa::FunctionEdgeType::FunctionCall:
    return FunctionCall;
  case efa::FunctionEdgeType::Invalid:
    return Invalid;
  case efa::FunctionEdgeType::DirectBranch:
    return DirectBranch;
  case efa::FunctionEdgeType::Return:
    return Return;
  case efa::FunctionEdgeType::BrokenReturn:
    return BrokenReturn;
  case efa::FunctionEdgeType::LongJmp:
    return LongJmp;
  case efa::FunctionEdgeType::Killer:
    return Killer;
  case efa::FunctionEdgeType::Unexpected:
    return Unexpected;
  case efa::FunctionEdgeType::Unreachable:
    return Unreachable;
  }

  revng_abort();
}

} // namespace yield::FunctionEdgeType

#include "revng/Yield/Generated/Late/FunctionEdgeType.h"
