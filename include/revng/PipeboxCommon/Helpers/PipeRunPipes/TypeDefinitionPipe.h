#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Progress.h"

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

    llvm::Task T1(3, "Running " + this->Name);
    T1.advance("prologue");

    T Instance(Model, this->StaticConfiguration, Configuration, Containers...);
    const model::Binary &Binary = *Model.get().get();
    auto &RequestedTypeDefinitions = Outgoing.at(this->OutputContainerIndex);

    T1.advance("run on type definitions", true);
    llvm::Task T2(RequestedTypeDefinitions.size(),
                  "Running pipe on type definitions");
    for (const ObjectID *Object : RequestedTypeDefinitions) {
      using TDK = model::TypeDefinition::Key;
      const TDK &Entry = std::get<TDK>(Object->key());
      T2.advance(toString(Entry), true);

      auto Committer = ODH.getCommitterFor(*Object, this->OutputContainerIndex);
      const UpcastablePointer<model::TypeDefinition>
        &TypeDefinition = Binary.TypeDefinitions().at(Entry);
      Instance.runOnTypeDefinition(TypeDefinition);
    }

    T1.advance("epilogue");
    return ODH.takeDependencies();
  }
};
