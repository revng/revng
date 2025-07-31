#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Concepts.h"
#include "revng/Pypeline/Model.h"

namespace pypeline::tracerunner {

class Container {
public:
  virtual ~Container() = default;

  virtual void
  deserialize(const std::map<const ObjectID *, llvm::ArrayRef<const char>>
                &Input) = 0;

  virtual std::map<ObjectID, pypeline::Buffer>
  serialize(llvm::ArrayRef<const ObjectID> ToSave) const = 0;

  virtual void *get() = 0;
};

class Analysis {
public:
  virtual ~Analysis() = default;

  virtual llvm::Error run(Model *TheModel,
                          std::vector<Container *> Containers,
                          pypeline::Request Incoming,
                          llvm::StringRef Configuration) = 0;
};

class Pipe {
public:
  virtual ~Pipe() = default;

  virtual pypeline::ObjectDependencies run(const Model *TheModel,
                                           std::vector<Container *> Containers,
                                           pypeline::Request Incoming,
                                           pypeline::Request Outgoing,
                                           llvm::StringRef Configuration) = 0;
};

class Registry {
private:
  using Container = tracerunner::Container;
  using Analysis = tracerunner::Analysis;
  using Pipe = tracerunner::Pipe;

  template<typename T, typename... Args>
  using ConstructorMap = llvm::StringMap<
    std::function<std::unique_ptr<T>(Args...)>>;

public:
  ConstructorMap<Container> Containers;
  ConstructorMap<Analysis> Analyses;
  ConstructorMap<Pipe, llvm::StringRef> Pipes;

public:
  Registry() {}
  Registry(const Registry &) = delete;
  Registry &operator=(const Registry &) = delete;
  Registry(Registry &&) = delete;
  Registry &operator=(const Registry &&) = delete;
};

} // namespace pypeline::tracerunner

namespace detail {

// Helper class to unpack containers from an ArrayRef
// To be used in conjunction with PipeRunner or AnalysisRunner
template<typename T>
class TraceRunnerInfo {
public:
  using Type = T;
  using ListType = llvm::ArrayRef<pypeline::tracerunner::Container *>;

  template<typename C, size_t I>
  static C unwrap(ListType Containers) {
    using Cptr = std::remove_reference_t<C> *;
    return *static_cast<Cptr>(Containers[I]->get());
  }
};

} // namespace detail
