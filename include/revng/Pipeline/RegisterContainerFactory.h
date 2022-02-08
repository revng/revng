#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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
