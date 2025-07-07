//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Helpers/Registrators.h"

#include "fakePipeboxImpl.h"

static RegisterContainer<StringContainer> X;
static RegisterPipe<AppendFooPipe> Y;
static RegisterAnalysis<AppendFooLibAnalysis> Z;
