#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/Registry.h"

namespace pipeline {

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

} // namespace pipeline
