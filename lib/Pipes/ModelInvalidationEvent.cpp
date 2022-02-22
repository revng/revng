/// \file ModelInvalidationEvent.cpp
/// \brief Implementation of model invalidation events

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/InvalidationEvent.h"
#include "revng/Pipeline/KindsRegistry.h"
#include "revng/Pipes/ModelInvalidationEvent.h"

using namespace pipeline;
using namespace revng::pipes;

char ModelInvalidationEvent::ID = 0;
