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
  static constexpr auto Name = "Compile";
  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(kinds::Root, 0, kinds::Object, 1) };
  }
  void run(const pipeline::Context &,
           pipeline::LLVMContainer &TargetsList,
           ObjectFileContainer &TargetBinary);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {
    OS << "llc " << ContainerNames[0] << " -o " << ContainerNames[1]
       << " --filetype=obj"
       << "\n";
  };
};

class CompileIsolatedModule {
public:
  static constexpr auto Name = "CompileIsolated";
  std::array<pipeline::ContractGroup, 1> getContract() const {
    pipeline::Contract RootPart(kinds::IsolatedRoot,
                                0,
                                kinds::Object,
                                1,
                                pipeline::InputPreservation::Preserve);
    pipeline::Contract IsolatedPart(kinds::Isolated, 0, kinds::Object, 1);
    return { pipeline::ContractGroup({ RootPart, IsolatedPart }) };
  }
  void run(const pipeline::Context &,
           pipeline::LLVMContainer &TargetsList,
           ObjectFileContainer &TargetBinary);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {
    OS << "llc " << ContainerNames[0] << " -o " << ContainerNames[1]
       << " --filetype=obj"
       << "\n";
  };
};

} // namespace revng::pipes
