#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

namespace pipeline {
class Context;
class Target;
class ContainerBase;
class Step;
struct PipeWrapper;

// A execution context is created and passed to each invocation of a pipe. It
// provides a reference to the pipeline::Context, as well as carrying around
// information about what is the state of the pipeline while the pipe is being
// executed, such as what is the current step.
//
// The execution context is the intended way of letting the pipeline know when a
// target is created, so that it may keep track of what paths inside pipeline
// globals have been read to produce such target and thus it can be invalidated
// when of element of the global indicated by such paths changes.
//
// We will not explain here how exactly the pipeline achieve this, but as a
// mental model what is happening is that every access to any field of the model
// is instrumented. When a ExecutionContext is created, or a when it commits (at
// the end of the call, just after the commit happens), the entire tracking
// state of the global is reset, and from that moment forward each time a
// variable is accessed it is marked as being accessed.
//
// When commit with arguments Target and Container is inovked, the following
// MUST be true:
// * Target must be == to the last target that has been created in Container
// * no field of any global can be read in betweet the last target being created
//   and Target being committed.
// * Target cannot be erased by Container after commit has been invoked.
// * No field of a global, directly or indirectly, can be read to produce a
//   target that is not the next Target to be committed. (that is, you can't
//   read fields to produce target X but then produce Y, commit Y, produce X,
//   commit X, unless you use pop and push to save and restore the state of read
//   fields).
//
// Less formally, you must produce and commit a target at the time.
//
// Example:
//
// void SomePipeProducingFunctions(const ExecutionContext& Ctx, SomeContainer&
// Container) {
//   ...
//   for (auto& Function : Container)
//   {
//     ...
//     Ctx.getContext().pushReadFields();
//     Ctx.commit(Container, Target(Function.metaadress()));
//     Ctx.getContext().popReadFields();
//   }
// }
//
class ExecutionContext {
private:
  Context *TheContext = nullptr;
  PipeWrapper *Pipe = nullptr;
  // false when running on a analysis
  bool RunningOnPipe = true;

  void commit(const Target &Target, llvm::StringRef ContainerName);

public:
  ~ExecutionContext();

public:
  ExecutionContext(Context &TheContext, PipeWrapper *Pipe);

public:
  void commit(const ContainerBase &Container, const Target &Target);
  void commitUniqueTarget(const ContainerBase &Container);

  void clearAndResumeTracking();

public:
  const Context &getContext() const { return *TheContext; }
  Context &getContext() { return *TheContext; }
};

} // namespace pipeline
