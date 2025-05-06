/// \file ExecutionContext.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/ExecutionContext.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Step.h"
#include "revng/Pipeline/Target.h"

using namespace pipeline;

ExecutionContext::ExecutionContext(Context &Context,
                                   PipeWrapper *Pipe,
                                   const ContainerToTargetsMap
                                     &RequestedTargets) :
  TheContext(&Context),
  Pipe(Pipe),
  Requested(RequestedTargets),
  RunningOnPipe(Pipe != nullptr) {
  // pipe is null when execution a analysis. We could just provide a context to
  // analyses, for the sake of uniformity we pass a execution context to them
  // too.
  if (RunningOnPipe)
    getContext().clearAndResume();
}

pipeline::ExecutionContext::~ExecutionContext() {
  if (RunningOnPipe)
    getContext().stopTracking();
}

void ExecutionContext::commit(const Target &Target,
                              llvm::StringRef ContainerName) {
  revng_assert(Pipe != nullptr);

  revng_log(InvalidationLog,
            "Committing " << Target.toString() << " into "
                          << ContainerName.str());

  // TODO: this triggers a sort at every invocation, which is bad for
  //       performance
  Committed.add(ContainerName.str(), Target);

  TargetInContainer ToCollect(Target, ContainerName.str());
  getContext().collectReadFields(ToCollect,
                                 Pipe->InvalidationMetadata.getPathCache());
}

void ExecutionContext::commitUniqueTarget(llvm::StringRef ContainerName) {
  revng_assert(getRequestedTargetsFor(ContainerName).size() == 1);
  commitAllFor(ContainerName);
}

void ExecutionContext::commitAllFor(llvm::StringRef ContainerName) {
  for (const pipeline::Target &Target : getRequestedTargetsFor(ContainerName))
    commit(Target, ContainerName);
}

void ExecutionContext::verify() const {
  if (not Requested.sameTargets(Committed)) {
    dbg << "After running the " << Pipe->Pipe->getName() << " pipe, ";
    dbg << "the list of requested targets is different from the committed "
           "ones.\n";
    dbg << "Requested:\n";
    Requested.dump();
    dbg << "Committed:\n";
    Committed.dump();
    revng_abort();
  }
}

char LoadExecutionContextPass::ID = '_';

template<typename T>
using RP = llvm::RegisterPass<T>;

static RP<LoadExecutionContextPass>
  X("load-execution-context-pass", "Load the execution context", true, true);
