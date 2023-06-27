//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Pass/AddPrimitiveTypes.h"
#include "revng/Model/Pass/AddPrimitiveTypesAnalysis.h"
#include "revng/Model/Type.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipeline/RegisterPipe.h"
#include "revng/Pipes/ModelGlobal.h"

namespace revng::pipes {

void AddPrimitiveTypesAnalysis::run(pipeline::ExecutionContext &Context) {
  model::addPrimitiveTypes(getWritableModelFromContext(Context));
}

} // end namespace revng::pipes

static pipeline::RegisterAnalysis<revng::pipes::AddPrimitiveTypesAnalysis> E;
