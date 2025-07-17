#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Concepts.h"
#include "revng/Pypeline/ModelImpl.h"
#include "revng/Pypeline/Utils.h"

namespace tracerunner {

class Container {
public:
  virtual ~Container() = default;

  virtual void *get() = 0;
};

class Analysis {
public:
  virtual ~Analysis() = default;

  virtual bool run(Model *,
                   std::vector<Container *> Containers,
                   detail::RequestT Incoming,
                   llvm::StringRef Configuration) = 0;
};

class Pipe {
public:
  virtual ~Pipe() = default;

  virtual detail::ObjectDependencies run(const Model *,
                                         std::vector<Container *> Containers,
                                         detail::RequestT Incoming,
                                         detail::RequestT Outgoing,
                                         llvm::StringRef Configuration) = 0;
};

} // namespace tracerunner

class TraceRunnerRegistry {
public:
  using Container = tracerunner::Container;
  using Analysis = tracerunner::Analysis;
  using Pipe = tracerunner::Pipe;

public:
  llvm::StringMap<std::function<std::unique_ptr<Container>()>> Containers;
  llvm::StringMap<std::function<std::unique_ptr<Analysis>()>> Analyses;
  llvm::StringMap<std::function<std::unique_ptr<Pipe>(llvm::StringRef)>> Pipes;

public:
  TraceRunnerRegistry() {}
  TraceRunnerRegistry(const TraceRunnerRegistry &) = delete;
  TraceRunnerRegistry &operator=(const TraceRunnerRegistry &) = delete;
  TraceRunnerRegistry(TraceRunnerRegistry &&) = delete;
  TraceRunnerRegistry &operator=(const TraceRunnerRegistry &&) = delete;
};
