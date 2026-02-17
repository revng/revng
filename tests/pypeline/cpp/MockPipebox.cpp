//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Helpers/Registrars.h"

#include "MockPipeboxImpl.h"

static RegisterContainer<StringContainer> X;
static RegisterContainer<RootStringContainer> X2;
static RegisterPipe<AppendFooPipe> Y;
static RegisterPipe<CreateEntryPipe> Y2;
static RegisterPipe<AppendFooPipe2> Y3;
static RegisterAnalysis<AppendFooLibAnalysis> Z;
static RegisterAnalysis<AppendFooLibAnalysis2> Z2;
