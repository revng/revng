#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Concepts.h"
#include "revng/Pypeline/Model.h"

namespace revng::pypeline::helpers::native {

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

class RegistryImpl {
private:
  template<typename T, typename... Args>
  using ConstructorMap = llvm::StringMap<std::unique_ptr<T> (*)(Args...)>;

public:
  ConstructorMap<Container> Containers;
  ConstructorMap<Analysis> Analyses;
  ConstructorMap<Pipe, llvm::StringRef> Pipes;

public:
  RegistryImpl() = default;
  ~RegistryImpl() = default;
  RegistryImpl(const RegistryImpl &) = delete;
  RegistryImpl &operator=(const RegistryImpl &) = delete;
  RegistryImpl(RegistryImpl &&) = delete;
  RegistryImpl &operator=(const RegistryImpl &&) = delete;
};

inline RegistryImpl Registry;

} // namespace revng::pypeline::helpers::native
