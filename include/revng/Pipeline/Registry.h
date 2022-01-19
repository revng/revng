#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Pipe.h"

namespace pipeline {

/// A registry is the general way of registering factories and kinds from down
/// stream libraries.
///
/// It is possible that a factory might require to look up something into the
/// context and thus this class used to that purpose.
///
/// For those kinds and containers and pipes that do not require to look inside
/// the loader look further into this file to fine more declarative classes.
class Registry {
protected:
  Registry() { getInstances().push_back(this); }

public:
  virtual ~Registry() = default;

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

public:
  virtual void registerContainersAndPipes(Loader &Loader) = 0;
  virtual void registerKinds(KindsRegistry &KindDictionary) = 0;
  virtual void libraryInitialization() = 0;
};

/// Instantiate a global object of this class for each kind you whish to
/// register
class RegisterKind : Registry {
private:
  Kind *K;

public:
  RegisterKind(Kind &K) : K(&K) {}

  ~RegisterKind() override = default;

public:
  void registerContainersAndPipes(Loader &) override {}

  void registerKinds(KindsRegistry &KindDictionary) override {
    KindDictionary.registerKind(*K);
  }

  void libraryInitialization() override {}
};

/// Instantiate a global object of this class for each pipe you wish to
/// register
template<typename PipeT>
class RegisterPipe : Registry {
private:
  llvm::StringRef Name;
  PipeT Pipe;

public:
  template<typename... Args>
  RegisterPipe(llvm::StringRef Name, Args &&...Arguments) :
    Name(Name), Pipe(std::forward<Args>(Arguments)...) {}

  template<typename... Args>
  RegisterPipe(Args &&...Arguments) requires HasName<PipeT>
    : Name(PipeT::Name), Pipe(std::forward<Args>(Arguments)...) {}

  ~RegisterPipe() override = default;

public:
  void registerContainersAndPipes(Loader &Loader) override {
    Loader.registerPipe(Name, Pipe);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}
};

/// Instantiate a global object of this class for each LLVMPass that you wish to
/// register
template<typename LLVMPass>
class RegisterLLVMPass : Registry {
private:
  llvm::StringRef Name;

public:
  RegisterLLVMPass(llvm::StringRef Name) : Name(Name) {}
  RegisterLLVMPass() : Name(LLVMPass::Name) {}

  ~RegisterLLVMPass() override = default;

public:
  void registerContainersAndPipes(Loader &Loader) override {
    Loader.registerLLVMPass<LLVMPass>(Name);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}

  void libraryInitialization() override {}
};

/// Instantiate a global object of this class for each factory you wish to
/// register
class RegisterContainerFactory : Registry {
private:
  llvm::StringRef Name;
  ContainerFactory Factory;

public:
  RegisterContainerFactory(llvm::StringRef Name, ContainerFactory Factory) :
    Name(Name), Factory(std::move(Factory)) {}

  ~RegisterContainerFactory() override = default;

public:
  void registerContainersAndPipes(Loader &Loader) override {
    Loader.addContainerFactory(Name, Factory);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}
};

/// Instantiate a global object of this class
/// for each default constructible container you wish to register
template<typename ContainerT>
class RegisterDefaultConstructibleContainer : Registry {
private:
  llvm::StringRef Name;

public:
  explicit RegisterDefaultConstructibleContainer(llvm::StringRef Name) :
    Name(Name) {}

  explicit RegisterDefaultConstructibleContainer() : Name(ContainerT::Name) {}

  ~RegisterDefaultConstructibleContainer() override = default;

public:
  void registerContainersAndPipes(Loader &Loader) override {
    Loader.addDefaultConstructibleContainer<ContainerT>(Name);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}
};

} // namespace pipeline
