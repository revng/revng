//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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
static RegisterContainer<BinariesContainer> C5;

//
// Pipes
//

using namespace revng::pypeline::pipes;
using namespace revng::pypeline::piperuns;

static RegisterSingleOutputPipe<Lift> P1;
static RegisterPipe<PureLLVMPassesRootPipe> P2;
static RegisterPipe<PureLLVMPassesPipe> P3;
