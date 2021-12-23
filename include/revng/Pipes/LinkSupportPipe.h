#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/LLVMContextWrapper.h"

namespace revng::pipes {

class LinkSupportPipe {
public:
  static constexpr auto Name = "LinkSupport";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(Root,
                                     pipeline::Exactness::DerivedFrom,
                                     0,
                                     pipeline::InputPreservation::Preserve) };
  }

  void run(const pipeline::Context &Ctx, pipeline::LLVMContainer &TargetsList);
};

} // namespace revng::pipes
