#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"

namespace revng::pipes {

class CompileModule {
public:
  static constexpr auto Name = "compile";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(kinds::Root, 0, kinds::Object, 1) };
  }
  void run(pipeline::ExecutionContext &,
           pipeline::LLVMContainer &ModuleContainer,
           ObjectFileContainer &TargetBinary);
};

class CompileIsolatedModule {
public:
  static constexpr auto Name = "compile-isolated";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    pipeline::Contract RootPart(kinds::IsolatedRoot,
                                0,
                                kinds::Object,
                                1,
                                pipeline::InputPreservation::Preserve);
    pipeline::Contract IsolatedPart(kinds::Isolated, 0, kinds::Object, 1);
    return { pipeline::ContractGroup({ RootPart, IsolatedPart }) };
  }

  void run(pipeline::ExecutionContext &,
           pipeline::LLVMContainer &ModuleContainer,
           ObjectFileContainer &TargetBinary);
};

} // namespace revng::pipes
