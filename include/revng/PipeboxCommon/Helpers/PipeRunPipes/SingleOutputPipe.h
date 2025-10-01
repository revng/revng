#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Base.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Helpers.h"
#include "revng/PipeboxCommon/Model.h"

template<typename T>
concept IsSingleOutputPipeRun = requires {
  requires IsSingleObjectPipeRun<T>;
  requires HasArgumentsDocumenation<T>;
  requires SpecializationOf<PipeRunContainerTypes<T>, TypeList>;
};

template<IsSingleOutputPipeRun T>
class SingleOutputPipe : public SingleOutputPipeBase<T> {
private:
  using Base = SingleOutputPipeBase<T>;

public:
  using ArgumentsDocumentation = T::ArgumentsDocumentation;

public:
  template<typename... Args>
    requires std::is_same_v<typename Base::ContainerTypes, TypeList<Args...>>
  revng::pypeline::ObjectDependencies
  run(const Model &Model,
      const revng::pypeline::Request &Incoming,
      const revng::pypeline::Request &Outgoing,
      llvm::StringRef Configuration,
      Args &...Containers) {
    revng_assert(Outgoing.at(this->OutputContainerIndex).size() == 1);

    ObjectDependenciesHelper ODH(Model, Outgoing, this->ContainerCount);
    T::run(Model, this->StaticConfiguration, Configuration, Containers...);
    ODH.commitUniqueTarget(this->OutputContainerIndex);
    return ODH.takeDependencies();
  }
};
