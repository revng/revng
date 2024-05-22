#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/Kinds.h"

namespace revng::pipes {

class LinkSupport {
public:
  static constexpr auto Name = "link-support";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(kinds::Root,
                                     0,
                                     pipeline::InputPreservation::Preserve) };
  }

  void run(const pipeline::ExecutionContext &Ctx,
           pipeline::LLVMContainer &TargetsList);

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const {
    return llvm::Error::success();
  }

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> RunningContainersNames) const;
};

} // namespace revng::pipes
