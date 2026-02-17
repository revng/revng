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

  virtual std::vector<ContainerArgument> signature() const = 0;
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

  virtual std::vector<ContainerArgument> signature() const override {
    std::vector<ContainerArgument> Result;

    using CT = PipeRunTraits<T>::ContainerTypes;
    forEach<CT>([&Result]<typename A, size_t I>() {
      using Argument = std::tuple_element_t<I, typename T::Arguments>;
      Result.push_back({ Argument::Name,
                         A::Name,
                         getEffectiveAccess<A>(Argument::Access) });
    });

    return Result;
  }

private:
  template<typename A>
  static Access getEffectiveAccess(const Access &Access) {
    if (Access != Access::Auto)
      return Access;

    if constexpr (std::is_const_v<A>)
      return Access::Read;
    else
      return Access::ReadWrite;
  }
};

} // namespace revng::pypeline::helpers::native
