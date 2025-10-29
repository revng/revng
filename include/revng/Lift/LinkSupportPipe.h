#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"
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

  void run(pipeline::ExecutionContext &EC,
           pipeline::LLVMContainer &ModuleContainer);
};

} // namespace revng::pipes

namespace revng::pypeline::piperuns {

class LinkSupport {
public:
  static constexpr llvm::StringRef Name = "LinkSupport";
  using Arguments = TypeList<
    PipeArgument<"Module", "Module to link support into">>;

public:
  static void run(const class Model &Model,
                  llvm::StringRef Config,
                  llvm::StringRef DynamicConfig,
                  LLVMRootContainer &ModuleContainer);
};

} // namespace revng::pypeline::piperuns
