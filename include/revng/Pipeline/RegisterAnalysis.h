#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Analysis.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Registry.h"

namespace pipeline {

/// Instantiate a global object of this class for each analysis you wish to
/// register
template<typename AnalsysisType>
class RegisterAnalysis : Registry {
private:
  llvm::StringRef Name;
  AnalsysisType Pipe;

public:
  template<typename... Args>
  RegisterAnalysis(llvm::StringRef Name, Args &&...Arguments) :
    Name(Name), Pipe(std::forward<Args>(Arguments)...) {}

  template<typename... Args>
  RegisterAnalysis(Args &&...Arguments) requires HasName<AnalsysisType>
    : Name(AnalsysisType::Name), Pipe(std::forward<Args>(Arguments)...) {}

  ~RegisterAnalysis() override = default;

public:
  void registerContainersAndPipes(Loader &Loader) override {
    Loader.registerAnalysis(Name, Pipe);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}
};

} // namespace pipeline
