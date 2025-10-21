//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/EarlyFunctionAnalysis/CollectCFG.h"
#include "revng/HeadersGeneration/ModelToHeaderPipe.h"
#include "revng/HeadersGeneration/ModelTypeDefinitionPipe.h"
#include "revng/Lift/Lift.h"
#include "revng/Pipebox/LLVMPipe.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/Helpers/Registrars.h"
#include "revng/PipeboxCommon/RawContainer.h"

using namespace revng::pypeline;

//
// Containers
//

static RegisterContainer<LLVMRootContainer> C1;
static RegisterContainer<LLVMFunctionContainer> C3;
static RegisterContainer<CBytesContainer> C4;
static RegisterContainer<BinariesContainer> C5;
static RegisterContainer<PTMLCTypeContainer> C6;
static RegisterContainer<CFGMap> C7;

//
// Pipes
//

using namespace revng::pypeline::pipes;
using namespace revng::pypeline::piperuns;

static RegisterSingleOutputPipe<Lift> P1;
static RegisterPipe<PureLLVMPassesRootPipe> P2;
static RegisterPipe<PureLLVMPassesPipe> P3;
static RegisterSingleOutputPipe<ModelToHeader> P4;
static RegisterTypeDefinitionPipe<GenerateModelTypeDefinition> P5;
static RegisterFunctionPipe<CollectCFG> P6;
