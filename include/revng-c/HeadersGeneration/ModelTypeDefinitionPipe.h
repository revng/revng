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

#include "revng-c/Pipes/Kinds.h"

namespace revng::pipes {

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

    return { ContractGroup({ Contract(kinds::Binary,
                                      0,
                                      ModelTypeDefinition,
                                      1,
                                      InputPreservation::Preserve) }) };
  }

  // Note: SourceBinary is not really needed, just a workaround.
  void run(pipeline::ExecutionContext &EC,
           const BinaryFileContainer &SourceBinary,
           ModelTypeDefinitionStringMap &ModelTypesContainer);
};

} // end namespace revng::pipes
