#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Base.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Helpers.h"

template<typename T>
concept IsFunctionPipe = requires(T &PipeRun, const model::Function &Function) {
  requires IsMultipleObjectsPipeRun<T>;
  requires hasConstructor<
    T,
    // The Structure of the constructor is:
    // {Model, StaticConfiguration, Configuration, Containers...}
    concat<TypeList<const Model &, llvm::StringRef, llvm::StringRef>,
           ConstructorContainerArguments<T>>>();
  { PipeRun.runOnFunction(Function) } -> std::same_as<void>;
};

template<IsFunctionPipe T>
class FunctionPipe : public SingleOutputPipeBase<T> {
private:
  using Base = SingleOutputPipeBase<T>;
  static_assert(Base::OutputContainerType::Kind == Kinds::Function);

public:
  using ArgumentsDocumentation = T::Arguments;

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
    T Instance(Model, this->StaticConfiguration, Configuration, Containers...);
    const model::Binary &Binary = *Model.get().get();

    for (const ObjectID *Object : Outgoing.at(this->OutputContainerIndex)) {
      auto Committer = ODH.getCommitterFor(*Object, this->OutputContainerIndex);

      const MetaAddress &Entry = std::get<MetaAddress>(Object->key());
      const model::Function &Function = Binary.Functions().at(Entry);
      Instance.runOnFunction(Function);
    }

    return ODH.takeDependencies();
  }
};
