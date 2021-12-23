/// \file Runner.cpp
/// \brief a runner top object of a pipeline structure, it is able to run the
/// pipeline.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"

#include "revng/Pipeline/Errors.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"

using namespace std;
using namespace llvm;
using namespace Pipeline;

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

  if (Requirements.empty()) {
    OS.changeColor(llvm::raw_ostream::Colors::GREEN);
    OS << "Already satisfied\n";
    return;
  }
  OS << "\n";

  OS.changeColor(llvm::raw_ostream::Colors::GREEN);
  OS << "Deduced Step Level Requirements: \n";

  for (const PipelineExecutionEntry &Entry : llvm::reverse(Requirements)) {
    if (Entry.Objectives.empty())
      continue;

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

    auto Deduced = S.deduceResults(Inputs);
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

  if (Splitted.size() != 3)
    return createStringError(inconvertibleErrorCode(),
                             "could not parse %s into three strings "
                             "step:container:inputfile",
                             ToParse.str().c_str());

  return PipelineFileMapping(Splitted[0], Splitted[1], Splitted[2]);
}

Error PipelineFileMapping::loadFromDisk(Runner &LoadInto) const {
  if (not LoadInto.containsStep(Step))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No known step " + Container);

  if (not LoadInto[Step].containers().containsOrCanCreate(Container))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No known container " + Container);

  return LoadInto[Step].containers()[Container].loadFromDisk(InputFile);
}

Error PipelineFileMapping::storeToDisk(const Runner &LoadInto) const {
  if (not LoadInto.containsStep(Step))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "No known step " + Container);

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

Error Runner::run(llvm::StringRef EndingStepName,
                  const ContainerToTargetsMap &Targets,
                  llvm::raw_ostream *DiagnosticLog) {
  optional<ContainerSet> CurrentContainer = std::nullopt;

  ContainerToTargetsMap ToLoad;
  vector<PipelineExecutionEntry> ToExec;

  if (auto
        Error = getObjectives(*this, EndingStepName, Targets, ToLoad, ToExec);
      Error)
    return Error;

  if (DiagnosticLog != nullptr)
    explainPipeline(Targets, ToLoad, ToExec, *DiagnosticLog);

  size_t CurrentStep = 0;
  for (auto &StepGoalsPairs : ToExec) {
    auto &[Step, Goals] = StepGoalsPairs;
    if (CurrentContainer.has_value()) {
      Step->containers().mergeBack(std::move(*CurrentContainer));
    }

    CurrentStep++;
    if (CurrentStep == ToExec.size())
      break;

    CurrentContainer = Step->cloneAndRun(*TheContext, ToLoad, DiagnosticLog);
    ToLoad = Step->deduceResults(CurrentContainer->enumerate());
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
    Out[NextStep.getName()].merge(Step.deduceResults(Out[Step.getName()]));
  }
}
