/// \file Runner.cpp
/// \brief a runner top object of a pipeline structure, it is able to run the
/// pipeline.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"

#include "revng/Pipeline/Errors.h"
#include "revng/Pipeline/GlobalTupleTreeDiff.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"

using namespace std;
using namespace llvm;
using namespace pipeline;

class PipelineExecutionEntry {
public:
  Step *ToExecute;
  ContainerToTargetsMap Objectives;

  PipelineExecutionEntry(Step &ToExecute, ContainerToTargetsMap Objectives) :
    ToExecute(&ToExecute), Objectives(std::move(Objectives)) {}
};

static Error getObjectives(Runner &Runner,
                           llvm::StringRef EndingStepName,
                           const ContainerToTargetsMap &Targets,
                           ContainerToTargetsMap &ToLoad,
                           std::vector<PipelineExecutionEntry> &ToExec) {
  ContainerToTargetsMap PartialGoals = Targets;
  auto *CurrentStep = &(Runner[EndingStepName]);
  while (CurrentStep != nullptr) {
    if (PartialGoals.empty())
      break;

    PartialGoals = CurrentStep->analyzeGoals(PartialGoals, ToLoad);
    ToExec.emplace_back(*CurrentStep, PartialGoals);
    CurrentStep = CurrentStep->hasPredecessor() ?
                    &CurrentStep->getPredecessor() :
                    nullptr;
  }

  if (not PartialGoals.empty())
    return make_error<UnsatisfiableRequestError>(Targets, PartialGoals);
  reverse(ToExec.begin(), ToExec.end());
  return Error::success();
}

using StatusMap = llvm::StringMap<ContainerToTargetsMap>;

static void explainPipeline(const ContainerToTargetsMap &Targets,
                            const ContainerToTargetsMap &ToLoad,
                            ArrayRef<PipelineExecutionEntry> Requirements,
                            raw_ostream &OS) {
  OS.changeColor(llvm::raw_ostream::Colors::GREEN);
  OS << "Objectives: \n";

  prettyPrintStatus(Targets, OS, 1);

  if (Requirements.size() <= 1) {
    OS.changeColor(llvm::raw_ostream::Colors::GREEN);
    OS << "Already satisfied\n";
    return;
  }
  OS << "\n";

  OS.changeColor(llvm::raw_ostream::Colors::GREEN);
  OS << "Deduced Step Level Requirements: \n";

  for (const PipelineExecutionEntry &Entry : llvm::reverse(Requirements)) {

    OS.indent(1);
    OS.changeColor(llvm::raw_ostream::Colors::MAGENTA);
    OS << Entry.ToExecute->getName() << "\n";

    prettyPrintStatus(Entry.Objectives, OS, 2);
  }

  OS.resetColor();
}

Error Runner::getInvalidations(StatusMap &Invalidated) const {

  for (const auto &NextS : *this) {
    if (not NextS.hasPredecessor())
      continue;

    const Step &S = NextS.getPredecessor();

    ContainerToTargetsMap &Inputs = Invalidated[S.getName()];

    ContainerToTargetsMap &Outputs = Invalidated[NextS.getName()];

    auto Deduced = NextS.deduceResults(Inputs);
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
  SmallVector<StringRef, 4> Splitted;
  ToParse.split(Splitted, ':', 2);

  if (Splitted.size() != 3) {
    auto *Message = "could not parse %s into three strings "
                    "step:container:inputfile";
    return createStringError(inconvertibleErrorCode(),
                             Message,
                             ToParse.str().c_str());
  }

  return PipelineFileMapping(Splitted[0], Splitted[1], Splitted[2]);
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

Error PipelineFileMapping::storeToDisk(const Runner &LoadInto) const {
  if (not LoadInto.containsStep(Step))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No known step " + Step);

  if (not LoadInto[Step].containers().containsOrCanCreate(Container))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No known container " + Container);

  return LoadInto[Step].containers().at(Container).storeToDisk(InputFile);
}

Error Runner::storeToDisk(llvm::StringRef DirPath) const {
  for (const auto &Step : Steps)
    if (auto Error = Step.second.storeToDisk(DirPath); !!Error)
      return Error;
  return TheContext->storeToDisk(DirPath);
}

Error Runner::loadFromDisk(llvm::StringRef DirPath) {
  if (auto Error = TheContext->loadFromDisk(DirPath); !!Error)
    return Error;
  for (auto &Step : Steps)
    if (auto Error = Step.second.loadFromDisk(DirPath); !!Error)
      return Error;
  return Error::success();
}

llvm::Expected<DiffMap>
Runner::runAnalysis(llvm::StringRef AnalysisName,
                    llvm::StringRef StepName,
                    const ContainerToTargetsMap &Targets,
                    llvm::raw_ostream *DiagnosticLog) {

  auto Before = getContext().getGlobals();

  auto MaybeStep = Steps.find(StepName);

  if (MaybeStep == Steps.end()) {
    return createStringError(inconvertibleErrorCode(),
                             "Could not find a step named %s\n",
                             StepName.str().c_str());
  }

  if (auto Error = run(StepName, Targets, DiagnosticLog))
    return std::move(Error);

  MaybeStep->second.runAnalysis(AnalysisName,
                                *TheContext,
                                Targets,
                                DiagnosticLog);

  auto &After = getContext().getGlobals();
  auto Map = Before.diff(After);
  for (const auto &Pair : Map)
    if (auto Error = apply(Pair.second))
      return std::move(Error);

  return std::move(Map);
}

/// Run all analysis in reverse post order (that is: parents first),
llvm::Expected<DiffMap> Runner::runAllAnalyses(llvm::raw_ostream *OS) {
  auto Before = getContext().getGlobals();

  for (const Step *Step : ReversePostOrderIndexes) {
    for (const auto &Pair : Step->analyses()) {
      const auto &Analysis = Pair.second;
      ContainerToTargetsMap Map;
      const auto &Containers = Analysis->getRunningContainersNames();
      for (size_t I = 0; I < Containers.size(); I++) {
        for (const Kind *K : Analysis->getAcceptedKinds(I)) {
          Map.add(Containers[I], Target(*K));
        }
      }

      auto Result = runAnalysis(Pair.first(), Step->getName(), Map, OS);
      if (not Result)
        return Result.takeError();
    }
  }

  auto &After = getContext().getGlobals();
  return Before.diff(After);
}

Error Runner::run(llvm::StringRef EndingStepName,
                  const ContainerToTargetsMap &Targets,
                  llvm::raw_ostream *DiagnosticLog) {

  ContainerToTargetsMap ToLoad;
  vector<PipelineExecutionEntry> ToExec;

  if (auto
        Error = getObjectives(*this, EndingStepName, Targets, ToLoad, ToExec);
      Error)
    return Error;

  if (DiagnosticLog != nullptr)
    explainPipeline(Targets, ToLoad, ToExec, *DiagnosticLog);

  if (ToExec.size() <= 1)
    return Error::success();

  auto &FirstStepContainers = ToExec.front().ToExecute->containers();
  auto CurrentContainer(FirstStepContainers.cloneFiltered(ToLoad));
  for (auto &StepGoalsPairs :
       llvm::make_range(ToExec.begin() + 1, ToExec.end())) {
    auto &[Step, Goals] = StepGoalsPairs;
    CurrentContainer = Step->cloneAndRun(*TheContext,
                                         std::move(CurrentContainer),
                                         DiagnosticLog);
  }

  if (DiagnosticLog != nullptr) {
    *DiagnosticLog << "Produced:\n";
    CurrentContainer.enumerate().dump(*DiagnosticLog);
  }

  return Error::success();
}

Error Runner::invalidate(const StatusMap &Invalidations) {
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
    Out[NextStep.getName()].merge(NextStep.deduceResults(Out[Step.getName()]));
  }
}
const KindsRegistry &Runner::getKindsRegistry() const {
  return TheContext->getKindsRegistry();
}

void Runner::getDiffInvalidations(const GlobalTupleTreeDiff &Diff,
                                  InvalidationMap &Map) const {
  for (const auto &Step : *this) {
    auto &StepInvalidations = Map[Step.getName()];
    for (const auto &Cotainer : Step.containers()) {
      if (not Cotainer.second)
        continue;

      auto &ContainerInvalidations = StepInvalidations[Cotainer.first()];
      for (const Kind &Rule : getKindsRegistry())
        Rule.getInvalidations(ContainerInvalidations, Diff);
    }
  }
}

llvm::Error Runner::apply(const GlobalTupleTreeDiff &Diff) {
  InvalidationMap Map;
  getDiffInvalidations(Diff, Map);
  if (auto Error = getInvalidations(Map); Error)
    return Error;
  return invalidate(Map);
}
