#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Helpers/Native/Container.h"
#include "revng/PipeboxCommon/Helpers/Native/Helpers.h"
#include "revng/PipeboxCommon/Helpers/PipeRunner.h"

namespace revng::pypeline::helpers::native {

class Pipe {
public:
  virtual ~Pipe() = default;

  virtual pypeline::ObjectDependencies run(const Model &TheModel,
                                           std::vector<Container *> Containers,
                                           const pypeline::Request &Incoming,
                                           const pypeline::Request &Outgoing,
                                           llvm::StringRef Configuration) = 0;
};

template<IsPipe T>
class PipeImpl final : public Pipe {
private:
  T Instance;

public:
  PipeImpl(llvm::StringRef Conf) : Instance(Conf) {}
  ~PipeImpl() override = default;

  virtual pypeline::ObjectDependencies
  run(const Model &TheModel,
      std::vector<Container *> Containers,
      const pypeline::Request &Incoming,
      const pypeline::Request &Outgoing,
      llvm::StringRef Configuration) override {
    return PipeRunner<ContainerListUnwrapper>::run(Instance,
                                                   TheModel,
                                                   Incoming,
                                                   Outgoing,
                                                   Configuration,
                                                   Containers);
  }
};

} // namespace revng::pypeline::helpers::native
