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
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/TupleTree/TupleTree.h"

namespace revng::pipes {

struct TaskArgument {};

struct PipeDescriptor {
  const char *Name;

  TaskArgument *Arguments;
  size_t ArgumentCount;

  void *Run;
  void *Invalidate;
  void *CheckPrecondition;
};

PipeDescriptor *revng_get_pipes(void);
size_t revng_get_pipe_count(void);
void revng_pipe_run(PipeDescriptor *Pipe, ...);
// void revng_pipe_invaldate(PipeDescriptor *Pipe);
void revng_pipe_check_precondition(PipeDescriptor *Pipe, void *Model);

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

  void run(pipeline::ExecutionContext &EC,
           const BinaryFileContainer &SourceBinary,
           pipeline::LLVMContainer &ModuleContainer);

  std::map<const pipeline::ContainerBase *, pipeline::TargetsList>
  invalidate(const BinaryFileContainer &SourceBinary,
             const pipeline::LLVMContainer &ModuleContainer,
             const pipeline::GlobalTupleTreeDiff &Diff) const;

  llvm::Error checkPrecondition(const pipeline::Context &Context) const;
};

static_assert(pipeline::HasCheckPrecondition<Lift>);

} // namespace revng::pipes
