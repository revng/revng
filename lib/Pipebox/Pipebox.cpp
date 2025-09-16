//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Helpers/Registrars.h"
#include "revng/PipeboxCommon/MapContainer.h"

using namespace revng::pypeline;

//
// Containers
//

static RegisterContainer<BytesContainer> C1;
static RegisterContainer<FunctionToBytesContainer> C2;
static RegisterContainer<TypeDefinitionToBytesContainer> C3;

//
// Pipes
//

using namespace revng::pypeline::pipes;
