#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Concepts.h"
#include "revng/Pypeline/Helpers/AnalysisRunner.h"
#include "revng/Pypeline/Helpers/Native/RunnerInfo.h"

namespace revng::pypeline::helpers::native {

template<IsAnalysis T>
class AnalysisImpl final : public Analysis {
private:
  T Instance;

public:
  AnalysisImpl() : Instance() {}
  ~AnalysisImpl() override = default;

  virtual llvm::Error run(Model *TheModel,
                          std::vector<Container *> Containers,
                          revng::pypeline::Request Incoming,
                          llvm::StringRef Configuration) override {
    using namespace revng::pypeline::helpers::native;
    return AnalysisRunner<RunnerInfo<T>>::run(Instance,
                                              &T::run,
                                              TheModel,
                                              Incoming,
                                              Configuration,
                                              Containers);
  }
};

} // namespace revng::pypeline::helpers::native
