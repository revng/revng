/// \file PipelineManager.cpp
/// \brief A pipeline manager ties up all the various bit and pieces of a
/// pipeline into a single object that does not require the c api to ever need
/// to expose a delete operator except for the global one.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <list>
#include <memory>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
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
using namespace std;
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
  const auto &ModelName = ModelGlobalName;
  class Context Ctx;

  Ctx.addGlobal<ModelGlobal>(ModelName);
  Ctx.addExternalContext("LLVMContext", Context);
  return Ctx;
}

static llvm::Error
pipelineConfigurationCallback(const Loader &Loader, LLVMPipe &NewPass) {
  using Wrapper = ModelGlobal;
  auto &Context = Loader.getContext();
  auto MaybeModelWrapper = Context.getGlobal<Wrapper>(ModelGlobalName);
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
  return MaybeMapping->loadFromDisk(*Runner);
}

llvm::Error PipelineManager::overrideModel(llvm::StringRef ModelOverride) {
  const auto &Name = ModelGlobalName;
  auto *Model(cantFail(PipelineContext->getGlobal<ModelGlobal>(Name)));
  return Model->loadFromDisk(ModelOverride);
}

static llvm::Expected<Runner>
setUpPipeline(pipeline::Context &PipelineContext,
              Loader &Loader,
              llvm::ArrayRef<std::string> TextPipelines,
              llvm::StringRef ExecutionDirectory) {
  auto MaybePipeline = Loader.load(TextPipelines);
  if (not MaybePipeline)
    return MaybePipeline.takeError();

  if (not ExecutionDirectory.empty())
    if (auto Error = MaybePipeline->loadFromDisk(ExecutionDirectory); Error)
      return std::move(Error);

  return MaybePipeline;
}

llvm::Expected<PipelineManager>
PipelineManager::create(llvm::ArrayRef<std::string> Pipelines,
                        llvm::ArrayRef<std::string> EnablingFlags,
                        llvm::StringRef ExecutionDirectory) {

  auto MaybeManager = createContexts(EnablingFlags, ExecutionDirectory);
  if (not MaybeManager)
    return MaybeManager.takeError();
  auto &Manager = *MaybeManager;

  std::vector<std::string> LoadedPipelines;

  for (const auto &Path : Pipelines) {
    auto MaybeBuffer = errorOrToExpected(MemoryBuffer::getFileOrSTDIN(Path));
    if (not MaybeBuffer)
      return MaybeBuffer.takeError();
    LoadedPipelines.emplace_back((*MaybeBuffer)->getBuffer().str());
  }
  if (auto MaybePipeline = setUpPipeline(*Manager.PipelineContext,
                                         *Manager.Loader,
                                         LoadedPipelines,
                                         ExecutionDirectory);
      MaybePipeline)
    Manager.Runner = make_unique<pipeline::Runner>(move(*MaybePipeline));
  else
    return MaybePipeline.takeError();

  return std::move(Manager);
}

llvm::Expected<PipelineManager>
PipelineManager::createContexts(llvm::ArrayRef<std::string> EnablingFlags,
                                llvm::StringRef ExecutionDirectory) {
  PipelineManager Manager;
  Manager.ExecutionDirectory = ExecutionDirectory.str();
  Manager.Context = std::make_unique<llvm::LLVMContext>();
  auto Ctx = setUpContext(*Manager.Context);
  Manager.PipelineContext = make_unique<pipeline::Context>(move(Ctx));

  auto Loader = setupLoader(*Manager.PipelineContext, EnablingFlags);
  Manager.Loader = make_unique<pipeline::Loader>(move(Loader));
  return Manager;
}

llvm::Expected<PipelineManager>
PipelineManager::createFromMemory(llvm::ArrayRef<std::string> PipelineContent,
                                  llvm::ArrayRef<std::string> EnablingFlags,
                                  llvm::StringRef ExecutionDirectory) {

  auto MaybeManager = createContexts(EnablingFlags, ExecutionDirectory);
  if (not MaybeManager)
    return MaybeManager.takeError();
  auto &Manager = *MaybeManager;
  if (auto MaybePipeline = setUpPipeline(*Manager.PipelineContext,
                                         *Manager.Loader,
                                         PipelineContent,
                                         ExecutionDirectory);
      MaybePipeline)
    Manager.Runner = make_unique<pipeline::Runner>(move(*MaybePipeline));
  else
    return MaybePipeline.takeError();

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

void PipelineManager::recalculateAllPossibleTargets() {
  CurrentState = Runner::State();
  getAllPossibleTargets(CurrentState);
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
      TargetsList Expansions;
      for (auto &Target : Container.second)
        Target.expand(*PipelineContext, Expansions);
      State[Step.first()][Container.first()] = std::move(Expansions);
    }
  }
}

void PipelineManager::getAllPossibleTargets(Runner::State &State) const {
  Runner->deduceAllPossibleTargets(State);
  for (auto &Step : State) {
    for (auto &Container : Step.second) {
      TargetsList Expansions;
      for (auto &Target : Container.second)
        Target.expand(*PipelineContext, Expansions);
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
      OS << Container.first() << ":\n";
      for (const auto &ExpandedTarget : Container.second)
        ExpandedTarget.dump(OS);
    }
  }
}

llvm::Error PipelineManager::storeToDisk() {
  if (ExecutionDirectory.empty())
    return llvm::Error::success();
  return Runner->storeToDisk(ExecutionDirectory);
}

llvm::Error PipelineManager::store(const PipelineFileMapping &Mapping) {
  return Mapping.storeToDisk(*Runner);
}

llvm::Error PipelineManager::overrideContainer(PipelineFileMapping Mapping) {
  return Mapping.loadFromDisk(*Runner);
}

llvm::Error
PipelineManager::store(llvm::ArrayRef<std::string> StoresOverrides) {
  for (const auto &Override : StoresOverrides) {
    auto MaybeMapping = PipelineFileMapping::parse(Override);
    if (not MaybeMapping)
      return MaybeMapping.takeError();

    if (auto Error = MaybeMapping->storeToDisk(*Runner))
      return Error;
  }
  return llvm::Error::success();
}

llvm::Error
PipelineManager::invalidateAllPossibleTargets(llvm::raw_ostream &Stream) {
  recalculateAllPossibleTargets();

  for (const auto &Step : CurrentState) {
    for (const auto &Container : Step.second) {
      for (const auto &Target : Container.second) {
        if (not getRunner()[Step.first()]
                  .containers()[Container.first()]
                  .enumerate()
                  .contains(Target))
          continue;

        Stream << "Invalidating: ";
        Stream << Step.first() << "/" << Container.first() << "/";
        Target.dump(Stream);
        InvalidationMap Map;
        Map[Step.first()][Container.first()].push_back(Target);
        if (auto Error = Runner->getInvalidations(Map); Error)
          return Error;
        if (auto Error = Runner->invalidate(Map); Error)
          return Error;

        for (const auto &First : Map)
          for (const auto &Second : First.second) {
            Stream << "\t" << First.first() << " " << Second.first() << " ";
            Target.dump(Stream);
          }
      }
    }
  }

  return llvm::Error::success();
}

llvm::Error
PipelineManager::produceAllPossibleTargets(llvm::raw_ostream &Stream) {
  recalculateAllPossibleTargets();

  for (const auto &Step : CurrentState) {
    for (const auto &Container : Step.second) {
      for (const auto &Target : Container.second) {
        ContainerToTargetsMap ToProduce;
        ToProduce.add(Container.first(), Target);
        Stream << Step.first() << "/" << Container.first() << "/";
        Target.dump(Stream);

        if (auto Error = Runner->run(Step.first(), ToProduce, &Stream); Error)
          return Error;
      }
    }
  }

  return llvm::Error::success();
}
