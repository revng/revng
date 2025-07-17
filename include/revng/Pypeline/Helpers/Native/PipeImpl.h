#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Helpers/Native/Registry.h"
#include "revng/Pypeline/Helpers/Native/RunnerInfo.h"
#include "revng/Pypeline/Helpers/PipeRunner.h"

namespace revng::pypeline::helpers::native {

template<IsPipe T>
class PipeImpl final : public Pipe {
private:
  T Instance;

public:
  PipeImpl(llvm::StringRef Conf) : Instance(Conf) {}
  ~PipeImpl() override = default;

  virtual pypeline::ObjectDependencies
  run(const Model *TheModel,
      std::vector<Container *> Containers,
      pypeline::Request Incoming,
      pypeline::Request Outgoing,
      llvm::StringRef Configuration) override {
    return PipeRunner<RunnerInfo<T>>::run(Instance,
                                          &T::run,
                                          TheModel,
                                          Incoming,
                                          Outgoing,
                                          Configuration,
                                          Containers);
  }
};

} // namespace revng::pypeline::helpers::native
