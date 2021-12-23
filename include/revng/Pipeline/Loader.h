#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <any>
#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMPipe.h"
#include "revng/Pipeline/PureLLVMPipe.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/SavableObject.h"
#include "revng/Support/Assert.h"
#include "revng/TupleTree/TupleTree.h"

namespace Pipeline {

/// this class is used to drive the grammar of pipelines yaml files
struct ContainerDeclaration {
  std::string Name;
  std::string Type;
};

/// this class is used to drive the grammar of pipelines yaml files
struct PipeInvocation {
  std::string Name;
  std::vector<std::string> UsedContainers;
  std::vector<std::string> Passes = {};
  std::vector<std::string> EnabledWhen = {};
};

/// this class is used to drive the grammar of pipelines yaml files
struct StepDeclaration {
  std::string Name;
  std::vector<PipeInvocation> Pipes;
  std::vector<std::string> EnabledWhen = {};
};

/// this class is used to drive the grammar of pipelines yaml files
struct PipelineDeclaration {
  std::string From;
  std::vector<ContainerDeclaration> Containers;
  std::vector<StepDeclaration> Steps;
};

class PipelineContext;

/// a loader is a class able to prodce a pipeline runner starting from
/// the yaml version of a pipeline.
///
/// the loader must be configured with appropriate containers and pipes bindings
/// if one wishes to register all those available one can simply use
/// Registry::registerAllContainersAndPipes
class Loader {
public:
  using LoaderCallback = std::function<llvm::Error(const Loader &, LLVMPipe &)>;

private:
  llvm::StringMap<ContainerFactory> KnownContainerTypes;
  llvm::StringMap<std::function<PipeWrapper(std::vector<std::string>)>>
    KnownPipesTypes;
  llvm::StringMap<std::function<std::unique_ptr<LLVMPassWrapperBase>()>>
    KnownLLVMPipeTypes;

  std::set<std::string> EnabledFlags;
  std::optional<LoaderCallback> OnLLVMContainerCreationAction = std::nullopt;
  Context *PipelineContext;

public:
  llvm::Expected<Runner> load(const PipelineDeclaration &) const;
  llvm::Expected<Runner> load(llvm::ArrayRef<PipelineDeclaration>) const;
  llvm::Expected<Runner> load(llvm::ArrayRef<std::string> Pipelines) const;

  explicit Loader(Context &C) : PipelineContext(&C) {}

  template<typename ContainerType>
  void addDefaultConstructibleContainer(llvm::StringRef Name) {
    auto [_,
          inserted] = KnownContainerTypes
                        .try_emplace(Name, [](llvm::StringRef ContainerName) {
                          return std::make_unique<ContainerType>(ContainerName);
                        });
    revng_assert(inserted);
  }

  void addContainerFactory(llvm::StringRef Name, ContainerFactory Factory) {
    KnownContainerTypes.try_emplace(Name, std::move(Factory));
  }

  template<typename LLVMPass>
  void registerLLVMPass(llvm::StringRef Name) {
    auto [_, inserted] = KnownLLVMPipeTypes.try_emplace(Name, []() {
      using Type = LLVMPassWrapper<LLVMPass>;
      return std::make_unique<Type>(LLVMPass());
    });

    revng_assert(inserted);
  }

  template<typename PipeType>
  void registerPipe(llvm::StringRef Name) {
    const auto LambdaToEmplace = [](std::vector<std::string> CName) {
      return PipeWrapper::makeWrapper<PipeType>(std::move(CName));
    };
    auto [_, inserted] = KnownPipesTypes.try_emplace(Name, LambdaToEmplace);
    revng_assert(inserted);
  }

  template<typename PipeType>
  void registerPipe(llvm::StringRef Name, const PipeType &Pipe) {
    const auto LambdaToEmplace =
      [Pipe](std::vector<std::string> ContainerNames) {
        return PipeWrapper(Pipe, std::move(ContainerNames));
      };
    auto [_, inserted] = KnownPipesTypes.try_emplace(Name, LambdaToEmplace);
    revng_assert(inserted);
  }

  void registerEnabledFlags(auto &NamesRange) {
    for (const auto &Name : NamesRange)
      EnabledFlags.insert(Name);
  }

  const Context &getContext() const { return *PipelineContext; }
  Context &getContext() { return *PipelineContext; }

  void setLLVMPipeConfigurer(LoaderCallback CallBack) {
    OnLLVMContainerCreationAction = std::move(CallBack);
  }

private:
  void emitTerminators(Runner &Runner) const;
  llvm::Error
  parseSteps(Runner &Runner, const PipelineDeclaration &Declaration) const;
  llvm::Error parseDeclarations(Runner &Runner,
                                const PipelineDeclaration &Declaration) const;

  llvm::Error parseStepDeclaration(Runner &Runner,
                                   const StepDeclaration &,
                                   std::string &LastAddedStep) const;
  llvm::Error
  parseInvocation(Step &Step, const PipeInvocation &Invocation) const;
  llvm::Error
  parseContainerDeclaration(Runner &Runner, const ContainerDeclaration &) const;

  llvm::Error parseLLVMPass(Step &Step, const PipeInvocation &Invocation) const;

  llvm::Error
  parsePureLLVMPass(Step &Step, const PipeInvocation &Invocation) const;

  bool isInvocationUsed(const std::vector<std::string> &Names) const;
};
} // namespace Pipeline

INTROSPECTION_NS(Pipeline, ContainerDeclaration, Name, Type);
template<>
struct llvm::yaml::MappingTraits<Pipeline::ContainerDeclaration>
  : public TupleLikeMappingTraits<Pipeline::ContainerDeclaration> {};

LLVM_YAML_IS_SEQUENCE_VECTOR(Pipeline::ContainerDeclaration)

template<>
struct llvm::yaml::MappingTraits<Pipeline::PipeInvocation> {
  static void mapping(IO &io, Pipeline::PipeInvocation &info) {
    io.mapRequired("Name", info.Name);
    io.mapRequired("UsedContainers", info.UsedContainers);
    io.mapOptional("Passes", info.Passes);
    io.mapOptional("EnabledWhen", info.EnabledWhen);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(Pipeline::PipeInvocation)

template<>
struct llvm::yaml::MappingTraits<Pipeline::StepDeclaration> {
  static void mapping(IO &io, Pipeline::StepDeclaration &info) {
    io.mapRequired("Name", info.Name);
    io.mapOptional("Pipes", info.Pipes);
    io.mapOptional("EnabledWhen", info.EnabledWhen);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(Pipeline::StepDeclaration)

INTROSPECTION_NS(Pipeline, PipelineDeclaration, Containers, Steps);
template<>
struct llvm::yaml::MappingTraits<Pipeline::PipelineDeclaration>
  : public TupleLikeMappingTraits<Pipeline::PipelineDeclaration> {
  static void mapping(IO &io, Pipeline::PipelineDeclaration &info) {
    io.mapOptional("From", info.From);
    io.mapRequired("Containers", info.Containers);
    io.mapRequired("Steps", info.Steps);
  }
};
