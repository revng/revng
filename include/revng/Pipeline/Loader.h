#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/Pipeline/Analysis.h"
#include "revng/Pipeline/ContainerFactory.h"
#include "revng/Pipeline/GenericLLVMPipe.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Support/Assert.h"
#include "revng/TupleTree/Introspection.h"
#include "revng/TupleTree/TupleTree.h"

namespace pipeline {

// The following classes are used to drive the grammar of pipelines YAML files
struct ContainerDeclaration {
  std::string Name;
  std::string Type;
  std::string Role = "";
};

struct AnalysisDeclaration {
  std::string Type;
  std::vector<std::string> UsedContainers;
  std::string Name = "";
  std::string Step = "";
};

struct PipeInvocation {
  std::string Type;
  std::vector<std::string> UsedContainers;
  std::vector<std::string> Passes = {};
  std::vector<std::string> EnabledWhen = {};
  std::string Name = "";
};

struct ArtifactsDeclaration {
  std::string Container;
  std::string Kind;
  std::string SingleTargetFilename;

  ArtifactsDeclaration() : Container(), Kind(), SingleTargetFilename() {}

  bool isValid() const {
    return !Container.empty() && !Kind.empty() && !SingleTargetFilename.empty();
  }
};

struct StepDeclaration {
  std::string Name;
  std::vector<PipeInvocation> Pipes;
  std::vector<std::string> EnabledWhen = {};
  ArtifactsDeclaration Artifacts = {};
  std::vector<AnalysisDeclaration> Analyses = {};
};

struct BranchDeclaration {
  std::string From;
  std::vector<StepDeclaration> Steps;
};

struct AnalysesListDeclaration {
  std::string Name;
  std::vector<std::string> UsedAnalysesNames;
};

struct PipelineDeclaration {
  std::string Component;
  std::vector<ContainerDeclaration> Containers;
  std::vector<BranchDeclaration> Branches;
  std::vector<AnalysisDeclaration> Analyses = {};
  std::vector<AnalysesListDeclaration> AnalysesLists = {};
};

class PipelineContext;

/// A Loader produces a pipeline runner starting from the YAML version of a
/// pipeline.
///
/// The loader must be configured with appropriate containers and pipes bindings
/// if one wishes to register all those available one can simply use
/// Registry::registerAllContainersAndPipes.
class Loader {
public:
  using LoaderCallback = std::function<llvm::Error(const Loader &, LLVMPipe &)>;

private:
  llvm::StringMap<ContainerFactory> KnownContainerTypes;
  llvm::StringMap<PipeWrapper> KnownPipesTypes;
  llvm::StringMap<AnalysisWrapper> KnownAnalysisTypes;
  llvm::StringMap<std::function<std::unique_ptr<LLVMPassWrapperBase>()>>
    KnownLLVMPipeTypes;

  llvm::StringMap<std::string> KnownContainerRoles;

  std::set<std::string> EnabledFlags;
  std::optional<LoaderCallback> OnLLVMContainerCreationAction = std::nullopt;
  Context *PipelineContext;

public:
  explicit Loader(Context &C) : PipelineContext(&C) {}

public:
  const Context &getContext() const { return *PipelineContext; }
  Context &getContext() { return *PipelineContext; }

public:
  llvm::Expected<Runner> load(const BranchDeclaration &) const;
  llvm::Expected<Runner> load(const PipelineDeclaration &) const;
  llvm::Expected<Runner> load(llvm::ArrayRef<PipelineDeclaration>) const;
  llvm::Expected<Runner> load(llvm::ArrayRef<std::string> Pipelines) const;

  template<typename ContainerType>
  void addDefaultConstructibleContainer(llvm::StringRef Name) {
    auto [_,
          inserted] = KnownContainerTypes
                        .try_emplace(Name,
                                     ContainerFactory::create<ContainerType>());
    revng_assert(inserted);
  }

  void addContainerFactory(llvm::StringRef Name, ContainerFactory Factory) {
    KnownContainerTypes.try_emplace(Name, std::move(Factory));
  }

  const ContainerFactory &
  getContainerFactory(llvm::StringRef ContainerName) const {
    return KnownContainerTypes.find(ContainerName)->second;
  }

  void addContainerRole(llvm::StringRef Name, llvm::StringRef Role) {
    KnownContainerRoles.try_emplace(Role, Name.str());
  }

  const llvm::StringMap<PipeWrapper> &getPipes() const {
    return KnownPipesTypes;
  }

  template<typename LLVMPass>
  void registerLLVMPass(llvm::StringRef Name) {
    auto [_, inserted] = KnownLLVMPipeTypes.try_emplace(Name, []() {
      using Type = LLVMPassWrapper<LLVMPass>;
      return std::make_unique<Type>(LLVMPass());
    });

    revng_assert(inserted);
  }

  template<typename AnalysisType>
  void registerAnalysis(llvm::StringRef Name, const AnalysisType &Analysis) {
    auto [_, inserted] = KnownAnalysisTypes
                           .try_emplace(Name,
                                        AnalysisWrapper::make(Analysis, {}));
    revng_assert(inserted);
  }

  template<typename PipeType>
  void registerPipe(llvm::StringRef Name) {
    auto [_, inserted] = KnownPipesTypes
                           .try_emplace(Name, PipeWrapper::make<PipeType>({}));
    revng_assert(inserted);
  }

  void registerEnabledFlags(auto &NamesRange) {
    for (const auto &Name : NamesRange)
      EnabledFlags.insert(Name);
  }

  void setLLVMPipeConfigurer(LoaderCallback CallBack) {
    OnLLVMContainerCreationAction = std::move(CallBack);
  }

public:
  const llvm::StringMap<PipeWrapper> &getRegisteredPipes() {
    return KnownPipesTypes;
  }

  const llvm::StringMap<AnalysisWrapper> &getRegisteredAnalyses() {
    return KnownAnalysisTypes;
  }

private:
  llvm::Error parseSteps(Runner &Runner,
                         const BranchDeclaration &Declaration,
                         const llvm::StringMap<std::string> &ReadOnlyNames,
                         const llvm::StringRef Component) const;
  llvm::Error
  parseDeclarations(Runner &Runner,
                    const PipelineDeclaration &Declaration,
                    llvm::StringMap<std::string> &ReadOnlyNames) const;

  llvm::Error
  parseStepDeclaration(Runner &Runner,
                       const StepDeclaration &,
                       std::string &LastAddedStep,
                       const llvm::StringMap<std::string> &ReadOnlyNames,
                       const llvm::StringRef Component) const;

  llvm::Expected<PipeWrapper>
  parseInvocation(Step &Step,
                  const PipeInvocation &Invocation,
                  const llvm::StringMap<std::string> &ReadOnlyNames) const;
  llvm::Expected<AnalysisWrapper>
  parseAnalysis(const AnalysisDeclaration &Declaration) const;
  llvm::Error
  parseContainerDeclaration(Runner &Runner,
                            const ContainerDeclaration &,
                            llvm::StringMap<std::string> &ReadOnlyNames) const;

  llvm::Expected<PipeWrapper>
  parseLLVMPass(const PipeInvocation &Invocation) const;

  llvm::Expected<std::unique_ptr<LLVMPassWrapperBase>>
  loadPassFromName(llvm::StringRef Name) const;

  bool isInvocationUsed(const std::vector<std::string> &Names) const;
};
} // namespace pipeline

INTROSPECTION_NS(pipeline, ContainerDeclaration, Name, Type, Role);
template<>
struct llvm::yaml::MappingTraits<pipeline::ContainerDeclaration> {
  static void mapping(IO &TheIO, pipeline::ContainerDeclaration &Info) {
    TheIO.mapRequired("Name", Info.Name);
    TheIO.mapRequired("Type", Info.Type);
    TheIO.mapOptional("Role", Info.Role);
  }
};

INTROSPECTION_NS(pipeline, AnalysesListDeclaration, Name, UsedAnalysesNames);
template<>
struct llvm::yaml::MappingTraits<pipeline::AnalysesListDeclaration> {
  static void mapping(IO &TheIO, pipeline::AnalysesListDeclaration &Info) {
    TheIO.mapRequired("Name", Info.Name);
    TheIO.mapOptional("Analyses", Info.UsedAnalysesNames);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(pipeline::ContainerDeclaration)

template<>
struct llvm::yaml::MappingTraits<pipeline::PipeInvocation> {
  static void mapping(IO &TheIO, pipeline::PipeInvocation &Info) {
    TheIO.mapRequired("Type", Info.Type);
    TheIO.mapRequired("UsedContainers", Info.UsedContainers);
    TheIO.mapOptional("Passes", Info.Passes);
    TheIO.mapOptional("EnabledWhen", Info.EnabledWhen);
    TheIO.mapOptional("Name", Info.Name);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(pipeline::PipeInvocation)

template<>
struct llvm::yaml::MappingTraits<pipeline::StepDeclaration> {
  static void mapping(IO &TheIO, pipeline::StepDeclaration &Info) {
    TheIO.mapRequired("Name", Info.Name);
    TheIO.mapOptional("Pipes", Info.Pipes);
    TheIO.mapOptional("EnabledWhen", Info.EnabledWhen);
    TheIO.mapOptional("Artifacts", Info.Artifacts);
    TheIO.mapOptional("Analyses", Info.Analyses);
  }
};

LLVM_YAML_IS_SEQUENCE_VECTOR(pipeline::StepDeclaration)
LLVM_YAML_IS_SEQUENCE_VECTOR(pipeline::AnalysisDeclaration)
LLVM_YAML_IS_SEQUENCE_VECTOR(pipeline::BranchDeclaration)
LLVM_YAML_IS_SEQUENCE_VECTOR(pipeline::AnalysesListDeclaration)

template<>
struct llvm::yaml::MappingTraits<pipeline::PipelineDeclaration> {
  static void mapping(IO &TheIO, pipeline::PipelineDeclaration &Info) {
    TheIO.mapRequired("Component", Info.Component);
    TheIO.mapOptional("Containers", Info.Containers);
    TheIO.mapOptional("Branches", Info.Branches);
    TheIO.mapOptional("Analyses", Info.Analyses);
    TheIO.mapOptional("AnalysesLists", Info.AnalysesLists);
  }
};

template<>
struct llvm::yaml::MappingTraits<pipeline::BranchDeclaration> {
  static void mapping(IO &TheIO, pipeline::BranchDeclaration &Info) {
    TheIO.mapOptional("From", Info.From);
    TheIO.mapRequired("Steps", Info.Steps);
  }
};

template<>
struct llvm::yaml::MappingTraits<pipeline::ArtifactsDeclaration> {
  static void mapping(IO &TheIO, pipeline::ArtifactsDeclaration &Info) {
    TheIO.mapRequired("Container", Info.Container);
    TheIO.mapRequired("Kind", Info.Kind);
    TheIO.mapRequired("SingleTargetFilename", Info.SingleTargetFilename);
  }
};

template<>
struct llvm::yaml::MappingTraits<pipeline::AnalysisDeclaration> {
  static void mapping(IO &TheIO, pipeline::AnalysisDeclaration &Info) {
    TheIO.mapRequired("Name", Info.Name);
    TheIO.mapRequired("Type", Info.Type);
    TheIO.mapOptional("Step", Info.Step);
    TheIO.mapRequired("UsedContainers", Info.UsedContainers);
  }
};
