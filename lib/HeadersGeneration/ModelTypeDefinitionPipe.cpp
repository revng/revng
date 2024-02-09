//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/PopulateTargetListContainer.h"
#include "revng/Pipes/StringMap.h"

#include "revng-c/Backend/DecompileFunction.h"
#include "revng-c/HeadersGeneration/ModelTypeDefinition.h"
#include "revng-c/HeadersGeneration/ModelTypeDefinitionPipe.h"
#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

using namespace pipeline;
static RegisterDefaultConstructibleContainer<TypeTargetList> F1;
static RegisterDefaultConstructibleContainer<ModelTypeDefinitionStringMap> F2;

inline constexpr char PipeName[] = "populate-type-kind-target-container";
using PopulateTypeKind = PopulateTargetListContainer<TypeTargetList, PipeName>;

using Container = ModelTypeDefinitionStringMap;
void GenerateModelTypeDefinition::run(const ExecutionContext &Ctx,
                                      TypeTargetList &TargetList,
                                      Container &ModelTypesContainer) {
  const model::Binary &Model = *getModelFromContext(Ctx);
  for (const pipeline::Target &Target : TargetList.getTargets()) {
    Container::KeyType
      Key = Container::keyFromString(Target.getPathComponents()[0]);
    ModelTypesContainer[Key] = dumpModelTypeDefinition(Model, Key);
  }
}

void GenerateModelTypeDefinition::print(const Context &Ctx,
                                        llvm::raw_ostream &OS,
                                        llvm::ArrayRef<std::string> Names)
  const {
  OS << "[CLI tools for pipes are deprecated]\n";
}

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::PopulateTypeKind> X1;
static pipeline::RegisterPipe<revng::pipes::GenerateModelTypeDefinition> X2;
