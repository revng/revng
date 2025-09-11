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
#include "llvm/Support/FormatVariadic.h"
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
#include "revng/Support/Assert.h"
#include "revng/Support/Chrono.h"
#include "revng/Support/ProgramRunner.h"
#include "revng/Support/ResourceFinder.h"
#include "revng/Support/TemporaryFile.h"

using namespace pipeline;
using namespace llvm;
using namespace ::revng::pipes;
using revng::FilePath;

static cl::opt<bool> CheckComponentsVersion("check-components-version",
                                            cl::desc("Delete container caches "
                                                     "if component hashes "
                                                     "don't match"),
                                            cl::init(false));

static cl::opt<bool> SaveAfterEveryAnalysis("save-after-every-analysis",
                                            cl::desc("Save to disk the context "
                                                     "after every analysis is "
                                                     "run"),
                                            cl::init(false));

class LoadModelPipePass {
private:
  ModelWrapper Wrapper;

public:
  static constexpr auto Name = "load-model";

  std::vector<ContractGroup> getContract() const { return {}; }

  explicit LoadModelPipePass(ModelWrapper Wrapper) :
    Wrapper(std::move(Wrapper)) {}

  void registerPasses(llvm::legacy::PassManager &Manager) {
    Manager.add(new LoadModelWrapperPass(Wrapper));
  }
};

static Context setUpContext(LLVMContext &Context) {
  const auto &ModelName = revng::ModelGlobalName;
  pipeline::Context TheContext;
  TheContext.addGlobal<revng::ModelGlobal>(ModelName);
  TheContext.addExternalContext("LLVMContext", Context);
  return TheContext;
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

llvm::Error
PipelineManager::purge(const revng::DirectoryPath &ExecutionDirectory) {
  std::vector<FilePath> FilePaths = Runner->getWrittenFiles(ExecutionDirectory);
  for (FilePath &Path : FilePaths) {
    auto MaybeExists = Path.exists();
    if (not MaybeExists)
      return MaybeExists.takeError();

    if (MaybeExists.get()) {
      if (llvm::Error Error = Path.remove())
        return Error;
    }
  }
  return llvm::Error::success();
}

llvm::Error PipelineManager::checkComponentsVersion(const revng::DirectoryPath
                                                      &ExecutionDirectory) {
  FilePath HashFile = ExecutionDirectory.getFile("components-hash");

  // check that the if the saved hash file exists
  auto MaybeExists = HashFile.exists();
  if (not MaybeExists)
    return MaybeExists.takeError();

  bool HashFileExists = MaybeExists.get();

  if (not CheckComponentsVersion)
    return HashFileExists ? HashFile.remove() : llvm::Error::success();

  std::string ActualHash = revng::getComponentsHash();
  std::string SavedHash;

  if (HashFileExists) {
    // Read the saved hash file
    auto MaybeReadableFile = HashFile.getReadableFile();
    if (not MaybeReadableFile)
      return MaybeReadableFile.takeError();

    SavedHash = MaybeReadableFile.get()->buffer().getBuffer().str();
  }

  // Check if the hashes match
  if (ActualHash == SavedHash)
    return llvm::Error::success();

  // First thing, remove the hash file, to guarantee that at worst then next
  // time we run we still trigger cleanup
  if (HashFileExists) {
    if (llvm::Error Error = HashFile.remove())
      return Error;
  }

  if (auto Error = purge(ExecutionDirectory))
    return Error;

  // Re-write the components-hash file, to avoid re-deleting files on
  // the next run
  auto MaybeWritableFile = HashFile.getWritableFile();
  if (not MaybeWritableFile)
    return MaybeWritableFile.takeError();

  MaybeWritableFile.get()->os() << ActualHash;
  return MaybeWritableFile.get()->commit();
}

static llvm::Expected<FilePath> getBackupFilePath(const FilePath &Path) {
  // Keep trying to get a non-occupied backup file path. The timestamp is
  // precise down to milliseconds, so it is low probability this takes
  // more than one iteration.
  while (true) {
    const uint64_t UnixTime = revng::getEpochInMilliseconds();
    auto BackupFilePath = Path.addExtension(std::to_string(UnixTime));
    if (not BackupFilePath.has_value())
      return revng::createError("Could not get backup file");

    auto MaybeExists = BackupFilePath->exists();

    if (not MaybeExists)
      return MaybeExists.takeError();

    if (not MaybeExists.get())
      return *BackupFilePath;
  }
}

static llvm::Error migrateModelFile(const FilePath &ModelFile) {
  TemporaryFile InputTemporaryFile("revng-model-migration-input", "yml");
  TemporaryFile OutputTemporaryFile("revng-model-migration-output", "yml");

  auto InputFile = FilePath::fromLocalStorage(InputTemporaryFile.path());
  if (auto Error = ModelFile.copyTo(InputFile); !!Error)
    return Error;

  {
    int ReturnCode = ::Runner.run("revng",
                                  { "model",
                                    "migrate",
                                    InputTemporaryFile.path().str(),
                                    "--output",
                                    OutputTemporaryFile.path().str() });

    if (ReturnCode != 0)
      return revng::createError("Error encountered while running migrations, "
                                "process returned: %d",
                                ReturnCode);
  }

  auto OutputFile = FilePath::fromLocalStorage(OutputTemporaryFile.path());
  if (auto Error = OutputFile.copyTo(ModelFile); !!Error)
    return Error;

  return llvm::Error::success();
}

/// Creates a backup for the model file and runs migrations on it.
/// \return On success returns the backup path. On failure returns the error and
/// leaves the model unchanged
static llvm::Expected<FilePath> migrateModel(const FilePath &ModelFile) {
  llvm::Expected<bool> ModelFileExists = ModelFile.exists();
  if (not ModelFileExists)
    return ModelFileExists.takeError();

  if (not ModelFileExists.get())
    revng_abort("Attempting to run migrations on a model file that does not "
                "exist");

  llvm::Expected<FilePath> ModelBackupFile = getBackupFilePath(ModelFile);
  if (not ModelBackupFile)
    return ModelBackupFile.takeError();

  if (auto Error = ModelFile.copyTo(ModelBackupFile.get()); !!Error)
    return Error;

  if (not ModelBackupFile)
    return ModelBackupFile.takeError();

  if (auto MigrationError = migrateModelFile(ModelFile); !!MigrationError) {
    // Migration failed, that's okay though, as the original model has not been
    // overridden (see migrateModelFile). Just return the error.
    return MigrationError;
  }

  return ModelBackupFile;
}

llvm::Error
PipelineManager::setUpPipeline(llvm::ArrayRef<std::string> TextPipelines) {
  auto MaybePipeline = Loader->load(TextPipelines);
  if (not MaybePipeline)
    return MaybePipeline.takeError();

  Runner = std::make_unique<pipeline::Runner>(std::move(MaybePipeline.get()));

  if (ExecutionDirectory.isValid()) {
    llvm::Error Error = checkComponentsVersion(ExecutionDirectory);
    if (Error)
      return Error;

    if (auto FirstLoadError = Runner
                                ->loadContextDirectory(ExecutionDirectory)) {
      // Loading failed, it could be due to an outdated model version, so we
      // perform model migrations and re-attempt to load. First check if there
      // is a model file.
      auto ModelFile = ExecutionDirectory.getDirectory("context")
                         .getFile(revng::ModelGlobalName);

      llvm::Expected<bool> ModelFileExists = ModelFile.exists();
      if (not ModelFileExists)
        return revng::joinErrors(ModelFileExists.takeError(),
                                 std::move(FirstLoadError));

      if (not ModelFileExists.get()) {
        // Model file does not exist, so nothing to migrate. Stop here and
        // return the first load error.
        return FirstLoadError;
      }

      llvm::Expected<FilePath> BackupFilePath = migrateModel(ModelFile);

      if (not BackupFilePath)
        return llvm::joinErrors(BackupFilePath.takeError(),
                                std::move(FirstLoadError));

      if (auto SecondLoadError = Runner
                                   ->loadContextDirectory(ExecutionDirectory)) {
        if (auto Error = BackupFilePath.get().copyTo(ModelFile)) {
          return revng::joinErrors(std::move(Error),
                                   std::move(SecondLoadError),
                                   std::move(FirstLoadError));
        }

        // Doesn't load even after migration, restore and report the errors?
        return llvm::joinErrors(std::move(SecondLoadError),
                                std::move(FirstLoadError));
      }

      // Load works after migration, so keep the migration result. Only thing
      // remaining is purging old containers.
      if (auto Error = purge(ExecutionDirectory))
        return Error;

      if (auto Error = StorageClient->commit())
        return Error;

      // All good, consume and log the FirstLoadError.
      {
        std::string Message = llvm::toString(std::move(FirstLoadError));
        revng_log(ExplanationLogger,
                  "First attempt at loading the context directory failed, but "
                  "it worked after migrations. Error: "
                    << Message);
      }
    }

    if (auto Error = Runner->loadContainers(ExecutionDirectory); !!Error)
      return Error;
  }

  Runner->resetDirtyness();

  return llvm::Error::success();
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
  LLVMContext = std::make_unique<llvm::LLVMContext>();
  auto Context = setUpContext(*LLVMContext);
  PipelineContext = make_unique<pipeline::Context>(std::move(Context));

  auto Loader = setupLoader(*PipelineContext, EnablingFlags);
  this->Loader = make_unique<pipeline::Loader>(std::move(Loader));
}

llvm::Expected<PipelineManager>
PipelineManager::createFromMemory(llvm::ArrayRef<std::string> PipelineContent,
                                  llvm::ArrayRef<std::string> EnablingFlags,
                                  llvm::StringRef ExecutionDirectory) {
  std::unique_ptr<revng::StorageClient> Client;
  if (not ExecutionDirectory.empty())
    Client = std::make_unique<revng::StorageClient>(ExecutionDirectory);
  return createFromMemory(PipelineContent, EnablingFlags, std::move(Client));
}

llvm::Expected<PipelineManager>
PipelineManager::createFromMemory(llvm::ArrayRef<std::string> PipelineContent,
                                  llvm::ArrayRef<std::string> EnablingFlags,
                                  std::unique_ptr<revng::StorageClient>
                                    &&Client) {
  PipelineManager Manager(EnablingFlags, std::move(Client));
  if (auto Error = Manager.setUpPipeline(PipelineContent))
    return Error;

  Manager.recalculateAllPossibleTargets();

  if (auto Error = Manager.computeDescription())
    return Error;

  if (Manager.ExecutionDirectory.isValid()) {
    if (auto Error = Manager.StorageClient->commit())
      return Error;
  }

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
  CurrentState = pipeline::Runner::State();
  getAllPossibleTargets(CurrentState, ExpandTargets);
  recalculateCache();
}

void PipelineManager::recalculateCurrentState() {
  CurrentState = pipeline::Runner::State();
  getCurrentState(CurrentState);
  recalculateCache();
}

void PipelineManager::getCurrentState(pipeline::Runner::State &State) const {
  Runner->getCurrentState(State);
  for (auto &Step : State) {
    for (auto &Container : Step.second) {
      State[Step.first()][Container.first()] = Container.second;
    }
  }
}

void PipelineManager::getAllPossibleTargets(pipeline::Runner::State &State,
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
  pipeline::Runner::State AvailableTargets;
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
  if (auto Error = Runner->store(ExecutionDirectory))
    return Error;

  // Commit all the changes to storage
  return StorageClient->commit();
}

llvm::Error PipelineManager::storeStepToDisk(llvm::StringRef StepName) {
  if (StorageClient == nullptr)
    return llvm::Error::success();

  auto &Step = Runner->getStep(StepName);
  if (auto Error = Runner->storeStepToDisk(StepName, ExecutionDirectory))
    return Error;
  return StorageClient->commit();
}

llvm::Expected<TargetInStepSet>
PipelineManager::deserializeContainer(pipeline::Step &Step,
                                      llvm::StringRef ContainerName,
                                      const llvm::MemoryBuffer &Buffer) {
  if (!Step.containers().isContainerRegistered(ContainerName))
    return revng::createError("Could not find container %s in step %s\n",
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
        if (auto Error = Runner->getInvalidations(Map))
          return std::move(Error);
        if (auto Error = Runner->invalidate(Map))
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

        if (auto Error = Runner->run(Step.first(), ToProduce))
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
                             TargetInStepSet &InvalidationsMap,
                             const llvm::StringMap<std::string> &Options,
                             llvm::raw_ostream *DiagnosticLog) {
  GlobalsMap Before = PipelineContext->getGlobals();

  Task T(List.size() + 1, "Analysis list " + List.getName());
  for (const AnalysisReference &Ref : List) {
    T.advance(Ref.getAnalysisName(), true);
    const Step &Step = Runner->getStep(Ref.getStepName());
    const AnalysisWrapper &Analysis = Step.getAnalysis(Ref.getAnalysisName());
    ContainerToTargetsMap Map;
    const std::vector<std::string>
      &Containers = Analysis->getRunningContainersNames();
    for (size_t I = 0; I < Containers.size(); I++) {
      for (const Kind *K : Analysis->getAcceptedKinds(I)) {
        Map.add(Containers[I], TargetsList::allTargets(*PipelineContext, *K));
      }
    }

    TargetInStepSet NewInvalidationsMap;
    auto Result = Runner->runAnalysis(Ref.getAnalysisName(),
                                      Step.getName(),
                                      Map,
                                      NewInvalidationsMap,
                                      Options);
    if (not Result)
      return Result.takeError();

    if (SaveAfterEveryAnalysis) {
      if (auto Error = store())
        return Error;
    }

    for (auto &NewEntry : NewInvalidationsMap)
      InvalidationsMap[NewEntry.first()].merge(NewEntry.second);
  }

  T.advance("Computing analysis list diff", true);

  recalculateAllPossibleTargets();
  DiffMap Diff = Before.diff(PipelineContext->getGlobals());

  PipelineContext->bumpCommitIndex();
  if (auto Error = store())
    return Error;

  return Diff;
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
  if (auto Error = store())
    return Error;

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
    return revng::createError("Step %s does not have any targets",
                              StepName.str().c_str());

  const auto &StepCurrentState = CurrentState[StepName];
  for (auto ContainerName : Map.keys()) {
    if (!StepCurrentState.contains(ContainerName))
      return revng::createError("Container %s does not have any targets",
                                ContainerName.str().c_str());

    auto &CurrentContainerState = StepCurrentState.at(ContainerName);
    for (const pipeline::Target &Target : Map.at(ContainerName)) {
      if (!CurrentContainerState.contains(Target))
        return revng::createError("Target %s cannot be produced",
                                  Target.toString().c_str());
    }
  }

  if (auto Error = getRunner().run(StepName, Map))
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

  if (auto Error = materializeTargets(StepName, Targets))
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

  if (auto Error = ExecutionDirectory.create())
    return Error;

  constexpr auto DescriptionName = "pipeline-description.yml";
  revng::FilePath DescriptionPath = ExecutionDirectory.getFile(DescriptionName);
  auto MaybeWritableFile = DescriptionPath.getWritableFile();
  if (!MaybeWritableFile)
    return MaybeWritableFile.takeError();

  MaybeWritableFile.get()->os() << this->Description;

  return MaybeWritableFile.get()->commit();
}
