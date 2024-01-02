/// \file PipelineManager.cpp
/// A pipeline manager ties up all the various bit and pieces of a pipeline into
/// a single object that does not require the c api to ever need to expose a
/// delete operator except for the global one.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <list>
#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Model/LoadModelPass.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/PipelineManager.h"
#include "revng/Support/ResourceFinder.h"

using namespace pipeline;
using namespace llvm;
using namespace ::revng::pipes;

class LoadModelPipePass {
private:
  ModelWrapper Wrapper;

public:
  static constexpr auto Name = "Load Model";
  std::vector<ContractGroup> getContract() const { return {}; }

  explicit LoadModelPipePass(ModelWrapper Wrapper) :
    Wrapper(std::move(Wrapper)) {}

  void registerPasses(llvm::legacy::PassManager &Manager) {
    Manager.add(new LoadModelWrapperPass(Wrapper));
  }

  // there is no need to print anything because the LoadModelPipePass is
  // implicitly added by revng opt.
  void print(llvm::raw_ostream &OS) const { OS << ""; }
};

static Context setUpContext(LLVMContext &Context) {
  const auto &ModelName = revng::ModelGlobalName;
  class Context Ctx;

  Ctx.addGlobal<revng::ModelGlobal>(ModelName);
  Ctx.addExternalContext("LLVMContext", Context);
  return Ctx;
}

static llvm::Error pipelineConfigurationCallback(const Loader &Loader,
                                                 LLVMPipe &NewPass) {
  using Wrapper = revng::ModelGlobal;
  auto &Context = Loader.getContext();
  auto MaybeModelWrapper = Context.getGlobal<Wrapper>(revng::ModelGlobalName);
  if (not MaybeModelWrapper)
    return MaybeModelWrapper.takeError();

  auto &Model = (*MaybeModelWrapper)->get();
  NewPass.emplacePass<LoadModelPipePass>(ModelWrapper(Model));
  return llvm::Error::success();
}

static Loader setupLoader(pipeline::Context &PipelineContext,
                          llvm::ArrayRef<std::string> EnablingFlags) {
  Loader Loader(PipelineContext);
  Loader.setLLVMPipeConfigurer(pipelineConfigurationCallback);
  Loader.registerEnabledFlags(EnablingFlags);
  Registry::registerAllContainersAndPipes(Loader);

  return Loader;
}

llvm::Error
PipelineManager::overrideContainer(llvm::StringRef PipelineFileMapping) {
  auto MaybeMapping = PipelineFileMapping::parse(PipelineFileMapping);
  if (not MaybeMapping)
    return MaybeMapping.takeError();
  return MaybeMapping->load(*Runner);
}

static llvm::Expected<Runner>
setUpPipeline(pipeline::Context &PipelineContext,
              Loader &Loader,
              llvm::ArrayRef<std::string> TextPipelines,
              const revng::DirectoryPath &ExecutionDirectory) {
  auto MaybePipeline = Loader.load(TextPipelines);
  if (not MaybePipeline)
    return MaybePipeline.takeError();

  if (ExecutionDirectory.isValid())
    if (auto Error = MaybePipeline->load(ExecutionDirectory); Error)
      return std::move(Error);

  return MaybePipeline;
}

llvm::Expected<PipelineManager>
PipelineManager::create(llvm::ArrayRef<std::string> Pipelines,
                        llvm::ArrayRef<std::string> EnablingFlags,
                        llvm::StringRef ExecutionDirectory) {
  std::vector<std::string> LoadedPipelines;

  std::vector<std::string> OrderedPipelines(Pipelines.begin(), Pipelines.end());
  llvm::sort(OrderedPipelines, [](std::string &Elem1, std::string &Elem2) {
    return llvm::sys::path::filename(Elem1) < llvm::sys::path::filename(Elem2);
  });

  for (const auto &Path : OrderedPipelines) {
    auto MaybeBuffer = errorOrToExpected(MemoryBuffer::getFileOrSTDIN(Path));
    if (not MaybeBuffer)
      return MaybeBuffer.takeError();
    LoadedPipelines.emplace_back((*MaybeBuffer)->getBuffer().str());
  }

  return createFromMemory(LoadedPipelines, EnablingFlags, ExecutionDirectory);
}

PipelineManager::PipelineManager(llvm::ArrayRef<std::string> EnablingFlags,
                                 std::unique_ptr<revng::StorageClient>
                                   &&Client) :
  StorageClient(std::move(Client)),
  ExecutionDirectory(StorageClient.get(), "") {
  Context = std::make_unique<llvm::LLVMContext>();
  auto Ctx = setUpContext(*Context);
  PipelineContext = make_unique<pipeline::Context>(std::move(Ctx));

  auto Loader = setupLoader(*PipelineContext, EnablingFlags);
  this->Loader = make_unique<pipeline::Loader>(std::move(Loader));
}

llvm::Expected<PipelineManager>
PipelineManager::createFromMemory(llvm::ArrayRef<std::string> PipelineContent,
                                  llvm::ArrayRef<std::string> EnablingFlags,
                                  llvm::StringRef ExecutionDirectory) {
  std::unique_ptr<revng::StorageClient> Client;
  if (not ExecutionDirectory.empty()) {
    auto MaybeClient = revng::StorageClient::fromPathOrURL(ExecutionDirectory);
    if (!MaybeClient)
      return MaybeClient.takeError();
    Client = std::move(MaybeClient.get());
  }

  return createFromMemory(PipelineContent, EnablingFlags, std::move(Client));
}

llvm::Expected<PipelineManager>
PipelineManager::createFromMemory(llvm::ArrayRef<std::string> PipelineContent,
                                  llvm::ArrayRef<std::string> EnablingFlags,
                                  std::unique_ptr<revng::StorageClient>
                                    &&Client) {
  PipelineManager Manager(EnablingFlags, std::move(Client));
  if (auto MaybePipeline = setUpPipeline(*Manager.PipelineContext,
                                         *Manager.Loader,
                                         PipelineContent,
                                         Manager.executionDirectory());
      MaybePipeline)
    Manager.Runner = make_unique<pipeline::Runner>(std::move(*MaybePipeline));
  else
    return MaybePipeline.takeError();

  Manager.recalculateAllPossibleTargets();

  if (auto Error = Manager.computeDescription(); Error)
    return Error;

  return std::move(Manager);
}

void PipelineManager::recalculateCache() {
  ContainerToEnumeration.clear();
  for (const auto &Step : *Runner) {
    for (const auto &Container : Step.containers()) {
      const auto &StepName = Step.getName();
      if (CurrentState.find(StepName) == CurrentState.end())
        continue;

      const auto &ContainerName = Container.first();
      if (not CurrentState[StepName].contains(ContainerName))
        continue;

      ContainerToEnumeration[&Container] = &CurrentState[StepName]
                                                        [ContainerName];
    }
  }
}

void PipelineManager::recalculateAllPossibleTargets(bool ExpandTargets) {
  CurrentState = Runner::State();
  getAllPossibleTargets(CurrentState, ExpandTargets);
  recalculateCache();
}

void PipelineManager::recalculateCurrentState() {
  CurrentState = Runner::State();
  getCurrentState(CurrentState);
  recalculateCache();
}

void PipelineManager::getCurrentState(Runner::State &State) const {
  Runner->getCurrentState(State);
  for (auto &Step : State) {
    for (auto &Container : Step.second) {
      State[Step.first()][Container.first()] = Container.second;
    }
  }
}

void PipelineManager::getAllPossibleTargets(Runner::State &State,
                                            bool ExpandTargets) const {
  Runner->deduceAllPossibleTargets(State);
  for (auto &Step : State) {
    for (auto &Container : Step.second) {
      TargetsList Expansions;
      for (auto &Target : Container.second) {
        Expansions.push_back(Target);
      }
      State[Step.first()][Container.first()] = std::move(Expansions);
    }
  }
}

void PipelineManager::writeAllPossibleTargets(llvm::raw_ostream &OS) const {
  Runner::State AvailableTargets;
  getAllPossibleTargets(AvailableTargets);
  for (const auto &Step : AvailableTargets) {

    OS << Step.first() << ":\n";
    for (const auto &Container : Step.second) {
      indent(OS, 1);
      OS << Container.first() << ":\n";
      for (const auto &ExpandedTarget : Container.second)
        ExpandedTarget.dump(OS, 2);
    }
  }
}

llvm::Error PipelineManager::store() {
  // If we are in ephemeral mode (resume was "") then we don't store anything
  if (StorageClient == nullptr)
    return llvm::Error::success();

  // Run store on the runner, this will serialize all step/containers
  // inside the resume directory
  if (auto Error = Runner->store(ExecutionDirectory); Error)
    return Error;

  // Commit all the changes to storage
  return StorageClient->commit();
}

llvm::Error PipelineManager::storeStepToDisk(llvm::StringRef StepName) {
  if (StorageClient == nullptr)
    return llvm::Error::success();

  auto &Step = Runner->getStep(StepName);
  if (auto Error = Runner->storeStepToDisk(StepName, ExecutionDirectory); Error)
    return Error;
  return StorageClient->commit();
}

llvm::Expected<TargetInStepSet>
PipelineManager::deserializeContainer(pipeline::Step &Step,
                                      llvm::StringRef ContainerName,
                                      const llvm::MemoryBuffer &Buffer) {
  if (!Step.containers().isContainerRegistered(ContainerName))
    return createStringError(inconvertibleErrorCode(),
                             "Could not find container %s in step %s\n",
                             ContainerName.str().c_str(),
                             Step.getName().str().c_str());

  auto &Container = Step.containers()[ContainerName];
  if (auto Error = Container.deserialize(Buffer); !!Error)
    return Error;

  auto MaybeInvalidations = invalidateAllPossibleTargets();
  if (not MaybeInvalidations)
    return MaybeInvalidations.takeError();

  if (auto Error = storeStepToDisk(Step.getName()); !!Error)
    return Error;

  PipelineContext->bumpCommitIndex();
  return MaybeInvalidations.get();
}

llvm::Error PipelineManager::store(const PipelineFileMapping &Mapping) {
  return Mapping.store(*Runner);
}

llvm::Error PipelineManager::overrideContainer(PipelineFileMapping Mapping) {
  return Mapping.load(*Runner);
}

llvm::Error
PipelineManager::store(llvm::ArrayRef<std::string> StoresOverrides) {
  for (const auto &Override : StoresOverrides) {
    auto MaybeMapping = PipelineFileMapping::parse(Override);
    if (not MaybeMapping)
      return MaybeMapping.takeError();

    if (auto Error = MaybeMapping->store(*Runner))
      return Error;
  }
  return llvm::Error::success();
}

llvm::Expected<TargetInStepSet>
PipelineManager::invalidateAllPossibleTargets() {
  TargetInStepSet ResultMap;
  auto Stream = ExplanationLogger.getAsLLVMStream();
  recalculateAllPossibleTargets();

  Task T(CurrentState.size(), "invalidateAllPossibleTargets");
  for (const auto &Step : CurrentState) {
    T.advance(Step.first(), true);
    if (Step.first() == Runner->begin()->getName())
      continue;

    Task T2(Step.second.size(), "Containers");
    for (const auto &Container : Step.second) {
      T2.advance(Container.first(), true);

      for (const auto &Target : Container.second) {
        auto &Containers = getRunner()[Step.first()].containers();
        if (not Containers.contains(Container.first()))
          continue;

        if (not Containers[Container.first()].enumerate().contains(Target))
          continue;

        *Stream << "Invalidating: ";
        *Stream << Step.first() << "/" << Container.first() << "/";
        Target.dump(*Stream);

        TargetInStepSet Map;
        Map[Step.first()][Container.first()].push_back(Target);
        if (auto Error = Runner->getInvalidations(Map); Error)
          return std::move(Error);
        if (auto Error = Runner->invalidate(Map); Error)
          return std::move(Error);

        for (const auto &First : Map) {
          for (const auto &Second : First.second) {
            *Stream << "\t" << First.first() << " " << Second.first() << " ";
            Target.dump(*Stream);
          }
        }

        pipeline::merge(ResultMap, Map);
      }
    }
  }

  return ResultMap;
}

llvm::Error PipelineManager::produceAllPossibleTargets(bool ExpandTargets) {
  recalculateAllPossibleTargets(ExpandTargets);

  for (const auto &Step : CurrentState) {
    for (const auto &Container : Step.second) {
      for (const auto &Target : Container.second) {
        ContainerToTargetsMap ToProduce;
        ToProduce.add(Container.first(), Target);
        ExplanationLogger << Step.first() << "/" << Container.first() << "/";
        auto Logger = ExplanationLogger.getAsLLVMStream();
        Target.dump(*Logger);
        ExplanationLogger << DoLog;

        if (auto Error = Runner->run(Step.first(), ToProduce); Error)
          return Error;
      }
    }
  }

  return llvm::Error::success();
}

const pipeline::Step::AnalysisValueType &
PipelineManager::getAnalysis(const AnalysisReference &Reference) const {
  auto &Step = getRunner().getStep(Reference.getStepName());

  auto Predicate = [&](const Step::AnalysisValueType &Analysis) -> bool {
    return Analysis.first() == Reference.getAnalysisName();
  };
  auto Analysis = llvm::find_if(Step.analyses(), Predicate);
  return *Analysis;
}

llvm::Expected<DiffMap>
PipelineManager::runAnalyses(const pipeline::AnalysesList &List,
                             TargetInStepSet &Map,
                             const llvm::StringMap<std::string> &Options,
                             llvm::raw_ostream *DiagnosticLog) {
  auto Result = Runner->runAnalyses(List, Map, Options);

  if (not Result)
    return Result.takeError();

  recalculateAllPossibleTargets();

  PipelineContext->bumpCommitIndex();
  return Result;
}

llvm::Expected<DiffMap>
PipelineManager::runAnalysis(llvm::StringRef AnalysisName,
                             llvm::StringRef StepName,
                             const ContainerToTargetsMap &Targets,
                             TargetInStepSet &Map,
                             const llvm::StringMap<std::string> &Options,
                             llvm::raw_ostream *DiagnosticLog) {
  auto Result = Runner->runAnalysis(AnalysisName,
                                    StepName,
                                    Targets,
                                    Map,
                                    Options);
  if (not Result)
    return Result.takeError();

  recalculateAllPossibleTargets();

  PipelineContext->bumpCommitIndex();
  return Result;
}

llvm::Expected<TargetInStepSet>
PipelineManager::invalidateFromDiff(const llvm::StringRef Name,
                                    const pipeline::GlobalTupleTreeDiff &Diff) {
  TargetInStepSet Map;
  if (auto ApplyError = getRunner().apply(Diff, Map); !!ApplyError)
    return std::move(ApplyError);

  // TODO: once invalidations are working, return `Map` instead of this
  return invalidateAllPossibleTargets();
}

llvm::Error
PipelineManager::materializeTargets(const llvm::StringRef StepName,
                                    const ContainerToTargetsMap &Map) {
  if (CurrentState.count(StepName) == 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Step %s does not have any targets",
                                   StepName.str().c_str());

  const auto &StepCurrentState = CurrentState[StepName];
  for (auto ContainerName : Map.keys()) {
    if (!StepCurrentState.contains(ContainerName))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Container %s does not have any targets",
                                     ContainerName.str().c_str());

    auto &CurrentContainerState = StepCurrentState.at(ContainerName);
    for (const pipeline::Target &Target : Map.at(ContainerName)) {
      if (!CurrentContainerState.contains(Target))
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Target %s cannot be produced",
                                       Target.serialize().c_str());
    }
  }

  if (auto Error = getRunner().run(StepName, Map); Error)
    return Error;

  return Error::success();
}

llvm::Expected<std::unique_ptr<pipeline::ContainerBase>>
PipelineManager::produceTargets(const llvm::StringRef StepName,
                                const Container &TheContainer,
                                const pipeline::TargetsList &List) {
  ContainerToTargetsMap Targets;
  for (const pipeline::Target &Target : List)
    Targets[TheContainer.second->name()].push_back(Target);

  if (auto Error = materializeTargets(StepName, Targets); Error)
    return Error;

  const auto &ToFilter = Targets.at(TheContainer.second->name());
  return TheContainer.second->cloneFiltered(ToFilter);
}

llvm::Error PipelineManager::computeDescription() {
  using pipeline::description::PipelineDescription;
  PipelineDescription Description = getRunner().description();

  {
    llvm::raw_string_ostream OS(this->Description);
    yaml::Output YAMLOutput(OS);
    YAMLOutput << Description;
  }

  if (StorageClient == nullptr)
    return llvm::Error::success();

  if (auto Error = ExecutionDirectory.create(); Error)
    return Error;

  constexpr auto DescriptionName = "pipeline-description.yml";
  revng::FilePath DescriptionPath = ExecutionDirectory.getFile(DescriptionName);
  auto MaybeWritableFile = DescriptionPath.getWritableFile();
  if (!MaybeWritableFile)
    return MaybeWritableFile.takeError();

  MaybeWritableFile.get()->os() << this->Description;

  return MaybeWritableFile.get()->commit();
}

llvm::Error
PipelineManager::setStorageCredentials(llvm::StringRef Credentials) {
  if (StorageClient == nullptr) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Client missing");
  }

  return StorageClient->setCredentials(Credentials);
}
