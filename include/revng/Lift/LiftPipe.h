#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/LegacyPassManager.h"

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMKind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/ResourceFinder.h"

namespace revng::pipes {

class Lift {
public:
  static constexpr auto Name = "lift";
  std::array<pipeline::ContractGroup, 1> getContract() const {
    return { pipeline::ContractGroup(kinds::Binary,
                                     0,
                                     kinds::Root,
                                     1,
                                     pipeline::InputPreservation::Preserve) };
  }

  void run(pipeline::ExecutionContext &Ctx,
           const BinaryFileContainer &SourceBinary,
           pipeline::LLVMContainer &ModuleContainer);

  std::map<const pipeline::ContainerBase *, pipeline::TargetsList>
  invalidate(const BinaryFileContainer &SourceBinary,
             const pipeline::LLVMContainer &ModuleContainer,
             const pipeline::GlobalTupleTreeDiff &Diff) const;

  llvm::Error checkPrecondition(const pipeline::Context &Ctx) const;

  void print(const pipeline::Context &Ctx,
             llvm::raw_ostream &OS,
             llvm::ArrayRef<std::string> ContainerNames) const {

    OS << *ResourceFinder.findFile("bin/revng");
    OS << " lift " << ContainerNames[0] << " " << ContainerNames[1] << "\n";
  }
};

} // namespace revng::pipes
