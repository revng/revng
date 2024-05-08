#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/Registry.h"

namespace pipeline {

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
public:
  ~RegisterDefaultConstructibleContainer() override = default;

  void registerContainersAndPipes(Loader &Loader) override {
    Loader.addDefaultConstructibleContainer<ContainerT>(ContainerT::Name);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}
};

/// Instantiate a global object of this class
/// for each role you wish to create.
class RegisterRole : Registry {
private:
  llvm::StringRef Name;
  llvm::StringRef Role;

public:
  RegisterRole(llvm::StringRef Name, llvm::StringRef Role) :
    Name(Name), Role(Role) {}

  ~RegisterRole() override = default;

public:
  void registerContainersAndPipes(Loader &Loader) override {
    Loader.addContainerRole(Name, Role);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}
};

} // namespace pipeline
