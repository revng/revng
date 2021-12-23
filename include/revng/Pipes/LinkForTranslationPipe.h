#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"

namespace revng::pipes {

class LinkForTranslationPipe {
public:
  static constexpr auto Name = "LinkForTranslation";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    pipeline::Contract BinaryPart(Binary,
                                  pipeline::Exactness::Exact,
                                  0,
                                  Translated,
                                  2,
                                  pipeline::InputPreservation::Preserve);
    pipeline::Contract ObjectPart(Object,
                                  pipeline::Exactness::Exact,
                                  1,
                                  Translated,
                                  2,
                                  pipeline::InputPreservation::Preserve);
    return { pipeline::ContractGroup({ BinaryPart, ObjectPart }) };
  }

  void run(const pipeline::Context &Ctx,
           FileContainer &InputBinary,
           FileContainer &ObjectFile,
           FileContainer &OutputBinary);
};

} // namespace revng::pipes
