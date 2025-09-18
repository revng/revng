//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Canonicalize/SimplifySwitchPass.h"
#include "revng/HeadersGeneration/ModelToHeaderPipe.h"
#include "revng/HeadersGeneration/ModelTypeDefinitionPipe.h"
#include "revng/PipeboxCommon/Helpers/Registrars.h"
#include "revng/PipeboxCommon/MapContainer.h"
#include "revng/Yield/Pipes/YieldAssembly.h"

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

static RegisterSingleOutputPipe<ModelToHeader> P1;
static RegisterFunctionPipe<YieldAssembly> P2;
static RegisterTypeDefinitionPipe<GenerateModelTypeDefinition> P3;
static RegisterLLVMFunctionPassPipe<SimplifySwitch> P4;
