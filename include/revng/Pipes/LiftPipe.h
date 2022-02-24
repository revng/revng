#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/LegacyPassManager.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"

namespace revng::pipes {

class LiftPipe {
public:
  static constexpr auto Name = "Lift";
  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(Binary,
                                     pipeline::Exactness::Exact,
                                     0,
                                     Root,
                                     1,
                                     pipeline::InputPreservation::Preserve) };
  }

  void run(pipeline::Context &Ctx,
           const FileContainer &SourceBinary,
           pipeline::LLVMContainer &TargetsList);

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {
    OS << "revng lift " << ContainerNames[0] << " -o " << ContainerNames[1]
       << "\n";
  }
};

} // namespace revng::pipes
