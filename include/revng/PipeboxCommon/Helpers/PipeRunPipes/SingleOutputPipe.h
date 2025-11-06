#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Progress.h"

#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Base.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Helpers.h"
#include "revng/PipeboxCommon/Model.h"

template<typename T>
concept IsSingleOutputPipeRun = requires {
  requires IsSingleObjectPipeRun<T>;
  requires HasArguments<T>;
  requires SpecializationOf<PipeRunContainerTypes<T>, TypeList>;
};

template<IsSingleOutputPipeRun T>
class SingleOutputPipe : public SingleOutputPipeBase<T> {
private:
  using Base = SingleOutputPipeBase<T>;

public:
  using Arguments = T::Arguments;

public:
  template<typename... Args>
    requires std::is_same_v<typename Base::ContainerTypes, TypeList<Args...>>
  revng::pypeline::ObjectDependencies
  run(const Model &Model,
      const revng::pypeline::Request &Incoming,
      const revng::pypeline::Request &Outgoing,
      llvm::StringRef Configuration,
      Args &...Containers) {
    ObjectDependenciesHelper ODH(Model, Outgoing, this->ContainerCount);
    auto &RequestedOutputs = Outgoing.at(this->OutputContainerIndex);
    if (RequestedOutputs.size() == 0)
      return ODH.takeDependencies();

    revng_assert(RequestedOutputs.size() == 1);
    llvm::Task T1(1, "Running " + this->Name);
    T1.advance("Running 'run'", true);
    T::run(Model, this->StaticConfiguration, Configuration, Containers...);
    ODH.commitUniqueTarget(this->OutputContainerIndex);
    return ODH.takeDependencies();
  }
};
