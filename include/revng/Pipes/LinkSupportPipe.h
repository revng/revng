#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Target.h"

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

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> RunningContainersNames) const;
};

} // namespace revng::pipes
