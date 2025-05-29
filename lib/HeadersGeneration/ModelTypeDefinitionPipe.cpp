//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Backend/DecompileFunction.h"
#include "revng/HeadersGeneration/PTMLHeaderBuilder.h"
#include "revng/Model/Binary.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/StringMap.h"

namespace revng::pipes {

inline constexpr char ModelTypeDefinitionMime[] = "text/x.c+tar+gz";
inline constexpr char ModelTypeDefinitionName[] = "model-type-definitions";
inline constexpr char ModelTypeDefinitionExtension[] = ".h";
using TypeDefinitionStringMap = TypeStringMap<&kinds::ModelTypeDefinition,
                                              ModelTypeDefinitionName,
                                              ModelTypeDefinitionMime,
                                              ModelTypeDefinitionExtension>;

class GenerateModelTypeDefinition {
public:
  static constexpr auto Name = "generate-model-type-definition";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(kinds::Binary,
                                      0,
                                      ModelTypeDefinition,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           const BinaryFileContainer &SourceBinary,
           TypeDefinitionStringMap &ModelTypesContainer) {
    const model::Binary &Model = *getModelFromContext(EC);
    for (const model::TypeDefinition &Type :
         getTypeDefinitionsAndCommit(EC, ModelTypesContainer.name())) {
      std::string &Result = ModelTypesContainer[Type.key()];
      llvm::raw_string_ostream Out(Result);

      ptml::CTypeBuilder B(Out,
                           Model,
                           true,
                           { .EnablePrintingOfTheMaximumEnumValue = true,
                             .EnableExplicitPaddingMode = false,
                             .EnableStructSizeAnnotation = true });

      B.printDefinition(Type);
    }
  }
};

using namespace pipeline;
static RegisterDefaultConstructibleContainer<TypeDefinitionStringMap> F2;

} // end namespace revng::pipes

static pipeline::RegisterPipe<revng::pipes::GenerateModelTypeDefinition> X2;
