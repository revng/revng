//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/RegisterContainerFactory.h"
#include "revng/Pipes/Containers.h"

using namespace revng::pipes;

static pipeline::RegisterDefaultConstructibleContainer<DecompileStringMap> R;
