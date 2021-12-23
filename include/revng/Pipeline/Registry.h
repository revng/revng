#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/Loader.h"

namespace Pipeline {

/// a registry is the general way of registering factories and kinds
/// from down stream libraries.
///
/// it is possible that a factory might require to look up something into the
/// context and thus this class used to that purpose
///
/// for those kinds and containers and pipes that do not require to look inside
/// the loader look further into this file to fine more declarative classes.
class Registry {
private:
  static llvm::SmallVector<Registry *, 4> &getInstances() {
    static llvm::SmallVector<Registry *, 4> Instances;
    return Instances;
  }

public:
  static void registerAllContainersAndPipes(Loader &Loader) {
    for (const auto &Reg : getInstances())
      Reg->registerContainersAndPipes(Loader);
  }

  static void runAllInitializationRoutines() {
    for (const auto &Reg : getInstances())
      Reg->libraryInitialization();
  }

  static KindsRegistry registerAllKinds() {
    KindsRegistry Registry;
    for (const auto &Reg : getInstances())
      Reg->registerKinds(Registry);

    Kind::init();
    Granularity::init();
    return Registry;
  }

  virtual ~Registry() = default;

  virtual void registerContainersAndPipes(Loader &Loader) = 0;
  virtual void registerKinds(KindsRegistry &KindDictionary) = 0;
  virtual void libraryInitialization() = 0;

protected:
  Registry() { getInstances().push_back(this); }
};

/// Instantiate a global object of this class for each
/// kind you whish to register.
class RegisterKind : Registry {
public:
  ~RegisterKind() override = default;

  RegisterKind(Kind &K) : K(&K) {}

  void registerContainersAndPipes(Loader &) override {}

  void registerKinds(KindsRegistry &KindDictionary) override {
    KindDictionary.registerKind(*K);
  }

  void libraryInitialization() override {}

private:
  Kind *K;
};

/// Instantiate a global object of this class for each
/// pipe you wish to register.
template<typename PipeT>
class RegisterPipe : Registry {
public:
  ~RegisterPipe() override = default;

  template<typename... Args>
  RegisterPipe(llvm::StringRef Name, Args &&...Arguments) :
    Name(Name), Pipe(std::forward<Args>(Arguments)...) {}

  void registerContainersAndPipes(Loader &Loader) override {
    Loader.registerPipe(Name, Pipe);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}

private:
  llvm::StringRef Name;
  PipeT Pipe;
};

/// Instantiate a global object of this class for each
/// llvmPipePass that you wish to register
template<typename LLVMPass>
class RegisterLLVMPass : Registry {
public:
  ~RegisterLLVMPass() override = default;

  RegisterLLVMPass(llvm::StringRef Name) : Name(Name) {}

  void registerContainersAndPipes(Loader &Loader) override {
    Loader.registerLLVMPass<LLVMPass>(Name);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}

private:
  llvm::StringRef Name;
};

/// Instantiate a global object of this class
/// for each factory you wish to register
class RegisterContainerFactory : Registry {
public:
  ~RegisterContainerFactory() override = default;

  RegisterContainerFactory(llvm::StringRef Name, ContainerFactory Factory) :
    Name(Name), Factory(std::move(Factory)) {}

  void registerContainersAndPipes(Loader &Loader) override {
    Loader.addContainerFactory(Name, Factory);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}

private:
  llvm::StringRef Name;
  ContainerFactory Factory;
};

/// Instantiate a global object of this class
/// for each default constructible container you wish to register
template<typename ContainerT>
class RegisterDefaultConstructibleContainer : Registry {
public:
  ~RegisterDefaultConstructibleContainer() override = default;

  explicit RegisterDefaultConstructibleContainer(llvm::StringRef Name) :
    Name(Name) {}

  void registerContainersAndPipes(Loader &Loader) override {
    Loader.addDefaultConstructibleContainer<ContainerT>(Name);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}

private:
  llvm::StringRef Name;
};

} // namespace Pipeline
