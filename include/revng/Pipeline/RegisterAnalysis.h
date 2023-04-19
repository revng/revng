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
template<typename AnalysisType>
class RegisterAnalysis : Registry {
private:
  llvm::StringRef Name;
  AnalysisType Pipe;
  std::vector<std::unique_ptr<CLOptionBase>> RegisteredOptions;

public:
  template<typename... Args>
  RegisterAnalysis(llvm::StringRef Name, Args &&...Arguments) :
    Name(Name),
    Pipe(std::forward<Args>(Arguments)...),
    RegisteredOptions(createCLOptions<AnalysisType>(&MainCategory)) {}

  template<typename... Args>
  RegisterAnalysis(Args &&...Arguments)
    requires HasName<AnalysisType>
    :
    Name(AnalysisType::Name),
    Pipe(std::forward<Args>(Arguments)...),
    RegisteredOptions(createCLOptions<AnalysisType>(&MainCategory)) {}

  ~RegisterAnalysis() override = default;

public:
  void registerContainersAndPipes(Loader &Loader) override {
    Loader.registerAnalysis(Name, Pipe);
  }

  void registerKinds(KindsRegistry &KindDictionary) override {}
  void libraryInitialization() override {}
};

} // namespace pipeline
