/// \file Runner.cpp
/// \brief a runner top object of a pipeline structure, it is able to run the
/// pipeline.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Errors.h"
#include "revng/Pipeline/GlobalTupleTreeDiff.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Assert.h"

using namespace std;
using namespace llvm;
using namespace pipeline;

class PipelineExecutionEntry {
public:
  Step *ToExecute;
  ContainerToTargetsMap Output;
  ContainerToTargetsMap Input;

  PipelineExecutionEntry(Step &ToExecute,
                         ContainerToTargetsMap Output,
                         ContainerToTargetsMap Input) :
    ToExecute(&ToExecute), Output(std::move(Output)), Input(std::move(Input)) {}
};

static Error getObjectives(Runner &Runner,
                           llvm::StringRef EndingStepName,
                           const ContainerToTargetsMap &Targets,
                           std::vector<PipelineExecutionEntry> &ToExec) {
  ContainerToTargetsMap ToLoad = Targets;
  auto *CurrentStep = &(Runner[EndingStepName]);
  while (CurrentStep != nullptr and not ToLoad.empty()) {

    ContainerToTargetsMap Output = ToLoad;
    ToLoad = CurrentStep->analyzeGoals(Runner.getContext(), ToLoad);
    ToExec.emplace_back(*CurrentStep, Output, ToLoad);
    CurrentStep = CurrentStep->hasPredecessor() ?
                    &CurrentStep->getPredecessor() :
                    nullptr;
  }

  if (not ToLoad.empty())
    return make_error<UnsatisfiableRequestError>(Targets, ToLoad);
  reverse(ToExec.begin(), ToExec.end());
  return Error::success();
}

static void explainPipeline(const ContainerToTargetsMap &Targets,
                            ArrayRef<PipelineExecutionEntry> Requirements) {
  if (Requirements.empty())
    return;

  ExplanationLogger << "OBJECTIVES requested\n";
  indent(ExplanationLogger, 1);

  if (Requirements.size() <= 1) {
    ExplanationLogger << "Already satisfied\n";
    return;
  }

  ExplanationLogger << Requirements.back().ToExecute->getName() << ":\n";
  prettyPrintStatus(Targets, ExplanationLogger, 2);

  ExplanationLogger << DoLog;
  ExplanationLogger << "DEDUCED steps content to be produced: \n";

  indent(ExplanationLogger, 1);
  ExplanationLogger << Requirements.back().ToExecute->getName() << ":\n";
  prettyPrintStatus(Targets, ExplanationLogger, 2);

  for (size_t I = Requirements.size(); I != 0; I--) {
    StringRef StepName = Requirements[I - 1].ToExecute->getName();
    const auto &TargetsNeeded = Requirements[I - 1].Input;

    indent(ExplanationLogger, 1);
    ExplanationLogger << StepName << ":\n";
    prettyPrintStatus(TargetsNeeded, ExplanationLogger, 2);
  }

  ExplanationLogger << DoLog;
}

Error Runner::getInvalidations(InvalidationMap &Invalidated) const {

  for (const auto &NextS : *this) {
    if (not NextS.hasPredecessor())
      continue;

    const Step &S = NextS.getPredecessor();

    ContainerToTargetsMap &Inputs = Invalidated[S.getName()];

    ContainerToTargetsMap &Outputs = Invalidated[NextS.getName()];

    auto Deduced = NextS.deduceResults(getContext(), Inputs);
    NextS.containers().intersect(Deduced);
    Outputs.merge(Deduced);
  }

  return Error::success();
}

Step &Runner::addStep(Step &&NewStep) {
  std::string Name = NewStep.getName().str();
  auto Info = Steps.try_emplace(Name, std::move(NewStep));
  revng_assert(Info.second);
  ReversePostOrderIndexes.emplace_back(&Info.first->second);
  return *ReversePostOrderIndexes.back();
}

llvm::Error Runner::getInvalidations(const Target &Target,
                                     InvalidationMap &Invalidations) const {
  for (const auto &Step : *this)
    for (const auto &Container : Step.containers()) {
      if (Container.second == nullptr)
        continue;

      if (Container.second->enumerate().contains(Target)) {
        Invalidations[Step.getName()].add(Container.first(), Target);
      }
    }
  if (auto Error = getInvalidations(Invalidations); !!Error)
    return Error;

  return llvm::Error::success();
}

Error Runner::invalidate(const Target &Target) {
  llvm::StringMap<ContainerToTargetsMap> Invalidations;
  if (auto Error = getInvalidations(Target, Invalidations); !!Error)
    return Error;
  return invalidate(Invalidations);
}

llvm::Expected<PipelineFileMapping>
PipelineFileMapping::parse(StringRef ToParse) {
  SmallVector<StringRef, 4> Parts;
  ToParse.split(Parts, ':', 2);

  if (Parts.size() != 2) {
    auto *Message = "could not parse %s into two parts "
                    "file_path:step/container";
    return createStringError(inconvertibleErrorCode(),
                             Message,
                             ToParse.str().c_str());
  }

  auto [StepName, ContainerName] = Parts.back().split('/');

  return PipelineFileMapping(StepName, ContainerName, Parts[0]);
}

Error PipelineFileMapping::loadFromDisk(Runner &LoadInto) const {
  if (not LoadInto.containsStep(Step))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No known step " + Step);

  if (not LoadInto[Step].containers().containsOrCanCreate(Container))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No known container " + Container);

  return LoadInto[Step].containers()[Container].loadFromDisk(InputFile);
}

Error PipelineFileMapping::storeToDisk(Runner &LoadInto) const {
  if (not LoadInto.containsStep(Step))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No known step " + Step);

  if (not LoadInto[Step].containers().containsOrCanCreate(Container))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No known container " + Container);

  return LoadInto[Step].containers()[Container].storeToDisk(InputFile);
}

Error Runner::storeToDisk(llvm::StringRef DirPath) const {
  for (const auto &StepName : Steps.keys()) {
    if (auto Error = storeStepToDisk(StepName, DirPath); !!Error) {
      return Error;
    }
  }

  llvm::SmallString<128> ContextDir;
  llvm::sys::path::append(ContextDir, DirPath, "context");
  if (auto EC = llvm::sys::fs::create_directories(ContextDir); EC)
    return llvm::createStringError(EC,
                                   "Could not create dir %s",
                                   ContextDir.c_str());

  return TheContext->storeToDisk(std::string(ContextDir));
}

Error Runner::storeStepToDisk(llvm::StringRef StepName,
                              llvm::StringRef DirPath) const {
  auto Step = Steps.find(StepName);
  if (Step == Steps.end())
    return createStringError(inconvertibleErrorCode(),
                             "Could not find a step named %s\n",
                             StepName.str().c_str());
  llvm::SmallString<128> StepDir;
  llvm::sys::path::append(StepDir, DirPath, Step->first());
  if (auto EC = llvm::sys::fs::create_directories(StepDir); EC)
    return llvm::createStringError(EC,
                                   "Could not create dir %s",
                                   StepDir.c_str());

  if (auto Error = Step->second.storeToDisk(std::string(StepDir)); !!Error)
    return Error;

  return Error::success();
}

Error Runner::loadFromDisk(llvm::StringRef DirPath) {
  llvm::SmallString<128> ContextDir;
  llvm::sys::path::append(ContextDir, DirPath, "context");
  if (auto Error = TheContext->loadFromDisk(std::string(ContextDir)); !!Error)
    return Error;

  for (auto &Step : Steps) {
    llvm::SmallString<128> StepDir;
    llvm::sys::path::append(StepDir, DirPath, Step.first());
    if (auto Error = Step.second.loadFromDisk(std::string(StepDir)); !!Error)
      return Error;
  }

  return Error::success();
}

llvm::Expected<DiffMap>
Runner::runAnalysis(llvm::StringRef AnalysisName,
                    llvm::StringRef StepName,
                    const ContainerToTargetsMap &Targets,
                    InvalidationMap &InvalidationsMap,
                    const llvm::StringMap<std::string> &Options) {

  auto Before = getContext().getGlobals();

  auto MaybeStep = Steps.find(StepName);

  if (MaybeStep == Steps.end()) {
    return createStringError(inconvertibleErrorCode(),
                             "Could not find a step named %s\n",
                             StepName.str().c_str());
  }

  if (auto Error = run(StepName, Targets))
    return std::move(Error);

  if (auto Error = MaybeStep->second.runAnalysis(AnalysisName,
                                                 *TheContext,
                                                 Targets,
                                                 Options);
      Error)
    return std::move(Error);

  auto &After = getContext().getGlobals();
  auto Map = Before.diff(After);
  for (const auto &GlobalNameDiffPair : Map)
    if (auto Error = apply(GlobalNameDiffPair.second, InvalidationsMap))
      return std::move(Error);

  return std::move(Map);
}

/// Run all analysis in reverse post order (that is: parents first),
llvm::Expected<DiffMap>
Runner::runAnalyses(const AnalysesList &List,
                    InvalidationMap &InvalidationsMap,
                    const llvm::StringMap<std::string> &Options) {
  auto Before = getContext().getGlobals();

  for (const AnalysisReference &Ref : List) {
    const auto &Step = getStep(Ref.getStepName());
    const auto &Analysis = Step.getAnalysis(Ref.getAnalysisName());
    ContainerToTargetsMap Map;
    const auto &Containers = Analysis->getRunningContainersNames();
    for (size_t I = 0; I < Containers.size(); I++) {
      for (const Kind *K : Analysis->getAcceptedKinds(I)) {
        Map.add(Containers[I], TargetsList::allTargets(getContext(), *K));
      }
    }

    auto Result = runAnalysis(Ref.getAnalysisName(),
                              Step.getName(),
                              Map,
                              InvalidationsMap,
                              Options);
    if (not Result)
      return Result.takeError();
  }

  auto &After = getContext().getGlobals();
  return Before.diff(After);
}

Error Runner::run(llvm::StringRef EndingStepName,
                  const ContainerToTargetsMap &Targets) {

  vector<PipelineExecutionEntry> ToExec;

  if (auto Error = getObjectives(*this, EndingStepName, Targets, ToExec); Error)
    return Error;

  explainPipeline(Targets, ToExec);

  if (ToExec.size() <= 1)
    return Error::success();

  for (auto &StepGoalsPairs : llvm::drop_begin(ToExec)) {
    auto &Step = *StepGoalsPairs.ToExecute;
    if (auto Error = Step.checkPrecondition(getContext()); Error)
      return llvm::make_error<AnnotatedError>(std::move(Error),
                                              "While scheduling step "
                                                + Step.getName() + ":");
  }

  for (auto &StepGoalsPairs : llvm::drop_begin(ToExec)) {
    auto &[Step, PredictedOutput, Input] = StepGoalsPairs;
    auto &Parent = Step->getPredecessor();
    auto CurrentContainer = Parent.containers().cloneFiltered(Input);
    Step->cloneAndRun(*TheContext, std::move(CurrentContainer));
    auto Produced = Step->containers().cloneFiltered(PredictedOutput);
    revng_check(Produced.enumerate().contains(PredictedOutput),
                "predicted output was not fully contained in actually "
                "produced");
    revng_check(Step->containers().enumerate().contains(PredictedOutput));
  }

  if (ExplanationLogger.isEnabled()) {
    ExplanationLogger << "PRODUCED \n";
    indent(ExplanationLogger, 1);
    ExplanationLogger << EndingStepName << ":\n";
    ToExec.back().ToExecute->containers().enumerate().dump(ExplanationLogger,
                                                           2,
                                                           false);
    ExplanationLogger << DoLog;
  }

  return Error::success();
}

Error Runner::invalidate(const InvalidationMap &Invalidations) {
  for (const auto &Step : Invalidations) {
    const auto &StepName = Step.first();
    const auto &ToRemove = Step.second;
    if (auto Error = operator[](StepName).invalidate(ToRemove); Error)
      return Error;
  }
  return Error::success();
}

void Runner::getCurrentState(State &Out) const {
  for (const auto &Step : Steps) {
    const auto &Under = Step.second;
    Out.insert({ Under.getName(), Under.containers().enumerate() });
  }
}

void Runner::deduceAllPossibleTargets(State &Out) const {
  getCurrentState(Out);

  for (const auto &NextStep : *this) {
    if (not NextStep.hasPredecessor())
      continue;

    const Step &Step = NextStep.getPredecessor();
    auto Result = NextStep.deduceResults(getContext(), Out[Step.getName()]);
    Out[NextStep.getName()].merge(std::move(Result));
  }
}
const KindsRegistry &Runner::getKindsRegistry() const {
  return TheContext->getKindsRegistry();
}

void Runner::getDiffInvalidations(const GlobalTupleTreeDiff &Diff,
                                  InvalidationMap &Map) const {
  // the custom write invalidation rules do not know what is the current
  // content of each container, so first we overestimate the targets to be
  // invalidated, and then we do the intersection between the overestimated one
  // and those that do exists.
  TargetsList OverestimatedTargets;
  for (const Kind &Kind : getKindsRegistry())
    Kind.getInvalidations(getContext(), OverestimatedTargets, Diff);

  for (const auto &Step : llvm::drop_begin(*this)) {
    auto &StepInvalidations = Map[Step.getName()];
    for (const auto &Container : Step.containers()) {
      if (not Container.second)
        continue;

      const TargetsList &ExisitingTargets = Container.second->enumerate();
      TargetsList NewTargets;
      for (const auto &Target : OverestimatedTargets)
        if (ExisitingTargets.contains(Target))
          NewTargets.emplace_back(Target);

      StepInvalidations[Container.first()] = NewTargets;
    }
  }
}

llvm::Error Runner::apply(const GlobalTupleTreeDiff &Diff,
                          InvalidationMap &Map) {
  getDiffInvalidations(Diff, Map);
  if (auto Error = getInvalidations(Map); Error)
    return Error;
  return invalidate(Map);
}
