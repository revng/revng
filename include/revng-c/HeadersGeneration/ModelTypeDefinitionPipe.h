#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/StringMap.h"
#include "revng/Pipes/TargetListContainer.h"

#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

inline constexpr char TypeContainerName[] = "type-kind-target-container";
using TypeTargetList = TargetListContainer<&kinds::ModelTypeDefinition,
                                           TypeContainerName>;

inline constexpr char ModelTypeDefinitionMime[] = "text/x.c+tar+gz";
inline constexpr char ModelTypeDefinitionName[] = "model-type-definitions";
inline constexpr char ModelTypeDefinitionExtension[] = ".h";
using ModelTypeDefinitionStringMap = TypeStringMap<
  &kinds::ModelTypeDefinition,
  ModelTypeDefinitionName,
  ModelTypeDefinitionMime,
  ModelTypeDefinitionExtension>;

class GenerateModelTypeDefinition {
public:
  static constexpr auto Name = "generate-model-type-definition";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(ModelTypeDefinition,
                                      0,
                                      ModelTypeDefinition,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  void run(const pipeline::ExecutionContext &Ctx,
           TypeTargetList &TargetList,
           ModelTypeDefinitionStringMap &ModelTypesContainer);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const;
};

} // end namespace revng::pipes
