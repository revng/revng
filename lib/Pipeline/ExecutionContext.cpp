/// \file ExecutionContext.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/ExecutionContext.h"
#include "revng/Pipeline/Step.h"
#include "revng/Pipeline/Target.h"

using namespace pipeline;

ExecutionContext::ExecutionContext(Context &Ctx,
                                   PipeWrapper *Pipe,
                                   const ContainerToTargetsMap
                                     &RequestedTargets) :
  TheContext(&Ctx),
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
  TargetInContainer ToCollect(Target, ContainerName.str());
  getContext().collectReadFields(ToCollect,
                                 Pipe->InvalidationMetadata.getPathCache());
}

void ExecutionContext::clearAndResumeTracking() {
  getContext().clearAndResume();
}

void ExecutionContext::commitUniqueTarget(const ContainerBase &Container) {
  auto Enumeration = Container.enumerate();
  revng_check(Enumeration.size() == 1);
  commit(Enumeration[0], Container.name());
}

void ExecutionContext::commit(const Target &Target,
                              const ContainerBase &Container) {
  commit(Target, Container.name());
}

char LoadExecutionContextPass::ID = '_';

template<typename T>
using RP = llvm::RegisterPass<T>;

static RP<LoadExecutionContextPass>
  X("load-execution-context-pass", "Load the execution context", true, true);
