#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/MapContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::pipes {

class GenerateModelTypeDefinition {
private:
  const Model &Model;
  TypeDefinitionToBytesContainer &Output;

public:
  static constexpr auto Name = "generate-model-type-definition";
  using Arguments = TypeList<
    PipeArgument<TypeDefinitionToBytesContainer, "Output", "">>;

  GenerateModelTypeDefinition(const class Model &Model,
                              llvm::StringRef Config,
                              llvm::StringRef DynamicConfig,
                              TypeDefinitionToBytesContainer &Output) :
    Model(Model), Output(Output) {}

  void runOnTypeDefinition(const UpcastablePointer<model::TypeDefinition>
                             &TypeDefinition);
};

} // namespace revng::pypeline::pipes
