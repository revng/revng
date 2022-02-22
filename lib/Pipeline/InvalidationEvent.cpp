/// \file InvalidationEvent.cpp
/// \brief Implementation of invalidation events

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/InvalidationEvent.h"

using namespace pipeline;

llvm::Error InvalidationEvent::run(Runner &Runner) const {
  Runner::InvalidationMap Map;
  getInvalidations(Runner, Map);
  if (auto Error = Runner.getInvalidations(Map); Error)
    return Error;
  return Runner.invalidate(Map);
}
