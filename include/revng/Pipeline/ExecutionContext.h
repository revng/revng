#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Generator.h"

namespace pipeline {
class Context;
class Target;
class ContainerBase;
class Step;
struct PipeWrapper;

template<typename F, typename T>
concept IsTargetToGlobal = requires {
  { std::is_same_v<F, const T &(const Target &)> };
};

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
  ContainerToTargetsMap Requested;
  ContainerToTargetsMap Committed;
  // false when running on a analysis
  bool RunningOnPipe = true;

public:
  ~ExecutionContext();

public:
  ExecutionContext(Context &TheContext,
                   PipeWrapper *Pipe,
                   const ContainerToTargetsMap &RequestedTargets = {});

public:
  void commit(const Target &Target, llvm::StringRef ContainerName);
  void commit(const Target &Target, const ContainerBase &Container) {
    commit(Target, Container.name());
  }

  void commitAllFor(llvm::StringRef ContainerName);
  void commitAllFor(const ContainerBase &Container) {
    commitAllFor(Container.name());
  }

  void commitUniqueTarget(llvm::StringRef ContainerName);
  void commitUniqueTarget(const ContainerBase &Container) {
    commitUniqueTarget(Container.name());
  }

  /// \note We do not provide a const version of this method so it doesn't get
  //        used accidentally in pipes.
  ContainerToTargetsMap &getCurrentRequestedTargets() { return Requested; }

  TargetsList &getRequestedTargetsFor(const ContainerBase &Container) {
    return getRequestedTargetsFor(Container.name());
  }

  TargetsList &getRequestedTargetsFor(llvm::StringRef ContainerName) {
    static TargetsList Empty;
    auto It = Requested.find(ContainerName);
    if (It == Requested.end())
      return Empty;
    else
      return It->second;
  }

  const TargetsList &
  getRequestedTargetsFor(const ContainerBase &Container) const {
    return getRequestedTargetsFor(Container.name());
  }

  const TargetsList &
  getRequestedTargetsFor(llvm::StringRef ContainerName) const {
    static TargetsList Empty;
    auto It = Requested.find(ContainerName);
    if (It == Requested.end())
      return Empty;
    else
      return It->second;
  }

  template<typename T, IsTargetToGlobal<T> F>
  cppcoro::generator<const T &>
  getAndCommit(const F &Extractor, llvm::StringRef ContainerName) {
    for (const Target &Target : getRequestedTargetsFor(ContainerName)) {
      getContext().pushReadFields();
      co_yield Extractor(Target);
      commit(Target, ContainerName);
      getContext().popReadFields();
    }
  }

public:
  /// Verifies all the requested targets have been committed
  void verify() const;

public:
  const Context &getContext() const { return *TheContext; }
  Context &getContext() { return *TheContext; }
};

class LoadExecutionContextPass : public llvm::ImmutablePass {
public:
  static char ID;

private:
  ExecutionContext *Ctx;
  llvm::StringRef ContainerName;

public:
  LoadExecutionContextPass(ExecutionContext *Ctx,
                           llvm::StringRef ContainerName) :
    llvm::ImmutablePass(ID), Ctx(Ctx), ContainerName(ContainerName) {}

  bool doInitialization(llvm::Module &M) override { return false; }

public:
  const TargetsList &getRequestedTargets() const {
    return Ctx->getRequestedTargetsFor(ContainerName);
  }
  llvm::StringRef getContainerName() const { return ContainerName; }
  ExecutionContext *get() { return Ctx; }
  const ExecutionContext *get() const { return Ctx; }
};

} // namespace pipeline
