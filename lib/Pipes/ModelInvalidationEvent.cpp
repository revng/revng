/// \file ModelInvalidationEvent.cpp
/// \brief Implementation of model invalidation events

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/InvalidationEvent.h"
#include "revng/Pipes/ModelInvalidationEvent.h"
#include "revng/Pipes/ModelInvalidationRule.h"

using namespace pipeline;
using namespace revng::pipes;

using InvalidationMap = Runner::InvalidationMap;

void ModelInvalidationEvent::getInvalidations(const Runner &Runner,
                                              InvalidationMap &Map) const {
  ModelInvalidationRule::registerAllInvalidations(Runner, Map);
}
