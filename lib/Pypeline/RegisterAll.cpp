//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/HeadersGeneration/ModelToHeaderPipe.h"
#include "revng/Pypeline/Helpers/Registrators.h"
#include "revng/Pypeline/MapContainer.h"

using namespace revng::pypeline;

//
// Containers
//

static RegisterContainer<RootBuffer> C1;
static RegisterContainer<FunctionMap> C2;
static RegisterContainer<TypeDefinitionMap> C3;

//
// Pipes
//

using namespace revng::pypeline::pipes;

static RegisterPipe<ModelToHeader> P1;
