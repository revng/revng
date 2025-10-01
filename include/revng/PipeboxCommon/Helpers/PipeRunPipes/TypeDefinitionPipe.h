#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Base.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/Helpers.h"

template<typename T>
concept IsTypeDefinitionPipe = requires(T &PipeRun,
                                        const UpcastablePointer<
                                          model::TypeDefinition> &TD) {
  requires IsMultipleObjectsPipeRun<T>;
  requires hasConstructor<
    T,
    // The Structure of the constructor is:
    // {Model, StaticConfiguration, Configuration, Containers...}
    concat<TypeList<const Model &, llvm::StringRef, llvm::StringRef>,
           ConstructorContainerArguments<T>>>();
  { PipeRun.runOnTypeDefinition(TD) } -> std::same_as<void>;
};

template<IsTypeDefinitionPipe T>
class TypeDefinitionPipe : public SingleOutputPipeBase<T> {
private:
  using Base = SingleOutputPipeBase<T>;
  static_assert(Base::OutputContainerType::Kind == Kinds::TypeDefinition);

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

      using TDK = model::TypeDefinition::Key;
      const TDK &Entry = std::get<TDK>(Object->key());
      const UpcastablePointer<model::TypeDefinition>
        &TypeDefinition = Binary.TypeDefinitions().at(Entry);
      Instance.runOnTypeDefinition(TypeDefinition);
    }
    return ODH.takeDependencies();
  }
};
