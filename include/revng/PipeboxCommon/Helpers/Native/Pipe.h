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

  virtual PipeOutput run(const Model &TheModel,
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

  virtual PipeOutput run(const Model &TheModel,
                         std::vector<Container *> Containers,
                         const pypeline::Request &Incoming,
                         const pypeline::Request &Outgoing,
                         llvm::StringRef Configuration) override {
    auto ContainerTuple = containerVectorToTuple<T>(Containers);
    return runPipe(Instance,
                   TheModel,
                   Incoming,
                   Outgoing,
                   Configuration,
                   ContainerTuple);
  }
};

} // namespace revng::pypeline::helpers::native
