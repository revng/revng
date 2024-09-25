//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/StringMap.h"

#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/HeadersGeneration/ModelTypeDefinition.h"
#include "revng-c/HeadersGeneration/ModelTypeDefinitionPipe.h"
#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

using namespace pipeline;
static RegisterDefaultConstructibleContainer<ModelTypeDefinitionStringMap> F2;

using Container = ModelTypeDefinitionStringMap;
void GenerateModelTypeDefinition::run(ExecutionContext &EC,
                                      const BinaryFileContainer &SourceBinary,
                                      Container &ModelTypesContainer) {
  const model::Binary &Model = *getModelFromContext(EC);
  for (const model::TypeDefinition &Type :
       getTypeDefinitionsAndCommit(EC, ModelTypesContainer.name())) {
    auto Key = Type.key();
    ModelTypesContainer[Key] = dumpModelTypeDefinition(Model, Key);
  }
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::GenerateModelTypeDefinition> X2;
