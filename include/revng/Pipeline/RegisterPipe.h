#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/CommandLine.h"

#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/Registry.h"

namespace pipeline {

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
  RegisterPipe(Args &&...Arguments)
    requires HasName<PipeT>
    : Name(PipeT::Name), Pipe(std::forward<Args>(Arguments)...) {}

  ~RegisterPipe() override = default;

public:
  void registerContainersAndPipes(Loader &Loader) override {
    Loader.registerPipe<PipeT>(Name);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}
};

} // namespace pipeline
