#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/IsolatedKind.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"

namespace revng::pipes {

class CompileModulePipe {
public:
  static constexpr auto Name = "Compile";
  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(Root,
                                     pipeline::Exactness::DerivedFrom,
                                     0,
                                     Object,
                                     1) };
  }
  void run(const pipeline::Context &,
           pipeline::LLVMContainer &TargetsList,
           FileContainer &TargetBinary);
};

class CompileIsolatedModulePipe {
public:
  static constexpr auto Name = "CompileIsolated";
  std::array<pipeline::ContractGroup, 1> getContract() const {
    pipeline::Contract RootPart(IsolatedRoot,
                                pipeline::Exactness::Exact,
                                0,
                                Object,
                                1,
                                pipeline::InputPreservation::Preserve);
    pipeline::Contract IsolatedPart(Isolated,
                                    pipeline::Exactness::Exact,
                                    0,
                                    Object,
                                    1);
    return { pipeline::ContractGroup({ RootPart, IsolatedPart }) };
  }
  void run(const pipeline::Context &,
           pipeline::LLVMContainer &TargetsList,
           FileContainer &TargetBinary);
};

} // namespace revng::pipes
