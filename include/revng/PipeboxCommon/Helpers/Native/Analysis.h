#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Helpers/AnalysisRunner.h"
#include "revng/PipeboxCommon/Helpers/Native/Container.h"
#include "revng/PipeboxCommon/Helpers/Native/Helpers.h"

namespace revng::pypeline::helpers::native {

class Analysis {
public:
  virtual ~Analysis() = default;

  virtual llvm::Error run(Model &TheModel,
                          std::vector<Container *> Containers,
                          const pypeline::Request &Incoming,
                          llvm::StringRef Configuration) = 0;
};

template<IsAnalysis T>
class AnalysisImpl final : public Analysis {
private:
  T Instance;

public:
  AnalysisImpl() : Instance() {}
  ~AnalysisImpl() override = default;

  virtual llvm::Error run(Model &TheModel,
                          std::vector<Container *> Containers,
                          const revng::pypeline::Request &Incoming,
                          llvm::StringRef Configuration) override {
    using namespace revng::pypeline::helpers::native;
    return AnalysisRunner<ContainerListUnwrapper>::run(Instance,
                                                       &T::run,
                                                       TheModel,
                                                       Incoming,
                                                       Configuration,
                                                       Containers);
  }
};

} // namespace revng::pypeline::helpers::native
