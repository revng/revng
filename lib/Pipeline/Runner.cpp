/// \file Runner.cpp
/// A runner top object of a pipeline structure, it is able to run the pipeline.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Progress.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Errors.h"
#include "revng/Pipeline/GlobalTupleTreeDiff.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Assert.h"
#include "revng/TupleTree/TupleTreeReference.h"

using namespace std;
using namespace llvm;
using namespace pipeline;

class PipelineExecutionEntry {
public:
  Step *ToExecute = nullptr;
  ContainerToTargetsMap Output;
  ContainerToTargetsMap Input;
  std::vector<PipeExecutionEntry> PipesExecuteEntries;

  PipelineExecutionEntry(Step &ToExecute,
                         ContainerToTargetsMap Output,
                         ContainerToTargetsMap Input,
                         std::vector<PipeExecutionEntry>
                           PipesToExecuteEntries) :
    ToExecute(&ToExecute),
    Output(std::move(Output)),
    Input(std::move(Input)),
    PipesExecuteEntries(std::move(PipesToExecuteEntries)) {}
};

static Error getObjectives(Runner &Runner,
                           llvm::StringRef EndingStepName,
                           const ContainerToTargetsMap &Targets,
                           std::vector<PipelineExecutionEntry> &ToExec) {
  ContainerToTargetsMap ToLoad = Targets;
  Step *CurrentStep = &(Runner[EndingStepName]);
  while (CurrentStep != nullptr and not ToLoad.empty()) {
    ContainerToTargetsMap Output = ToLoad;
    auto &&[Required,
            PipesExecutionEntries] = CurrentStep->analyzeGoals(ToLoad);
    ToLoad = std::move(Required);
    ToExec.emplace_back(*CurrentStep,
                        Output,
                        ToLoad,
                        std::move(PipesExecutionEntries));
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

  ExplanationLogger << "Requested targets:\n";
  indent(ExplanationLogger, 1);

  if (Requirements.size() <= 1) {
    ExplanationLogger << "Already satisfied\n";
    return;
  }

  ExplanationLogger << Requirements.back().ToExecute->getName() << ":\n";
  prettyPrintStatus(Targets, ExplanationLogger, 2);

  ExplanationLogger << DoLog;

  ExplanationLogger << "We need to have the following targets at the beginning "
                       "of the steps\n";

  indent(ExplanationLogger, 1);
  ExplanationLogger << Requirements.back().ToExecute->getName() << ":\n";
  prettyPrintStatus(Targets, ExplanationLogger, 2);

  for (size_t I = Requirements.size(); I != 0; I--) {
    StringRef StepName = Requirements[I - 1].ToExecute->getName();
    const ContainerToTargetsMap &TargetsNeeded = Requirements[I - 1].Input;

    indent(ExplanationLogger, 1);
    ExplanationLogger << StepName << ":\n";
    prettyPrintStatus(TargetsNeeded, ExplanationLogger, 2);
  }

  ExplanationLogger << DoLog;
}

Error Runner::getInvalidations(TargetInStepSet &Invalidated) const {

  for (const Step &NextS : *this) {
    if (not NextS.hasPredecessor())
      continue;

    const Step &S = NextS.getPredecessor();

    ContainerToTargetsMap &Inputs = Invalidated[S.getName()];

    ContainerToTargetsMap &Outputs = Invalidated[NextS.getName()];

    ContainerToTargetsMap Deduced = NextS.deduceResults(Inputs);
    NextS.containers().intersect(Deduced);
    Outputs.merge(Deduced);
  }

  return Error::success();
}

Step &Runner::addStep(Step &&NewStep) {
  std::string Name = NewStep.getName().str();
  auto Info = Steps.try_emplace(Name, std::move(NewStep));
  revng_assert(Info.second, ("Duplicate_step: " + Name).c_str());
  ReversePostOrderIndexes.emplace_back(&Info.first->second);
  return *ReversePostOrderIndexes.back();
}

llvm::Error Runner::getInvalidations(const Target &Target,
                                     TargetInStepSet &Invalidations) const {
  for (const Step &Step : *this)
    for (const auto &Container : Step.containers()) {
      if (Container.second == nullptr)
        continue;

      if (Container.second->enumerate().contains(Target)) {
        Invalidations[Step.getName()].add(Container.first(), Target);
      }
    }
  if (llvm::Error Error = getInvalidations(Invalidations); !!Error)
    return Error;

  return llvm::Error::success();
}

Error Runner::invalidate(const Target &Target) {
  llvm::StringMap<ContainerToTargetsMap> Invalidations;
  if (llvm::Error Error = getInvalidations(Target, Invalidations); !!Error)
    return Error;
  return invalidate(Invalidations);
}

llvm::Expected<PipelineFileMapping>
PipelineFileMapping::parse(StringRef ToParse) {
  if (ToParse.count(':') < 1 or ToParse.count('/') < 1) {
    auto *Message = "could not parse %s\n"
                    "Format is: file_path:step/container";
    return revng::createError(Message, ToParse.str().c_str());
  }

  auto &&[StoragePath, ContainerPath] = ToParse.rsplit(':');
  auto Path = revng::FilePath::fromLocalStorage(StoragePath);

  auto &&[StepName, ContainerName] = ContainerPath.split('/');
  return PipelineFileMapping(StepName, ContainerName, std::move(Path));
}

Error PipelineFileMapping::load(Runner &LoadInto) const {
  if (not LoadInto.containsStep(Step))
    return revng::createError("No known step " + Step);

  if (not LoadInto[Step].containers().containsOrCanCreate(Container))
    return revng::createError("No known container " + Container);

  return LoadInto[Step].containers()[Container].load(Path);
}

Error PipelineFileMapping::store(Runner &LoadInto) const {
  if (not LoadInto.containsStep(Step)) {
    return revng::createError("No known step " + Step);
  }

  if (not LoadInto[Step].containers().containsOrCanCreate(Container)) {
    return revng::createError("No known container " + Container);
  }

  return LoadInto[Step].containers()[Container].store(Path);
}

Error Runner::storeContext(const revng::DirectoryPath &DirPath) const {
  revng::DirectoryPath ContextDir = DirPath.getDirectory("context");
  if (auto Error = ContextDir.create())
    return Error;

  return TheContext->store(ContextDir);
}

Error Runner::store(const revng::DirectoryPath &DirPath) const {
  if (auto Error = DirPath.create())
    return Error;

  for (const auto &StepName : Steps.keys()) {
    if (llvm::Error Error = storeStepToDisk(StepName, DirPath); !!Error) {
      return Error;
    }
  }
  return storeContext(DirPath);
}

Error Runner::storeStepToDisk(llvm::StringRef StepName,
                              const revng::DirectoryPath &DirPath) const {
  auto Step = Steps.find(StepName);
  if (Step == Steps.end())
    return revng::createError("Could not find a step named %s\n",
                              StepName.str().c_str());

  revng::DirectoryPath StepDir = DirPath.getDirectory(Step->first());
  if (auto Error = StepDir.create())
    return Error;

  if (auto Error = Step->second.store(StepDir); !!Error)
    return Error;

  return Error::success();
}

Error Runner::load(const revng::DirectoryPath &DirPath) {
  revng::DirectoryPath ContextDir = DirPath.getDirectory("context");
  if (auto Error = TheContext->load(ContextDir); !!Error)
    return Error;

  for (auto &Step : Steps) {
    revng::DirectoryPath StepDir = DirPath.getDirectory(Step.first());
    if (auto Error = Step.second.load(StepDir); !!Error)
      return Error;
  }

  return Error::success();
}

std::vector<revng::FilePath>
Runner::getWrittenFiles(const revng::DirectoryPath &DirPath) const {
  std::vector<revng::FilePath> Result;

  for (const llvm::StringMapEntry<Step> &StepEntry : Steps) {
    // Exclude the "begin" step from written files (since it's written
    // manually by the user)
    if (StepEntry.first() == begin()->getName())
      continue;

    revng::DirectoryPath StepDir = DirPath.getDirectory(StepEntry.first());
    const pipeline::Step &Step = StepEntry.second;
    append(Step.getWrittenFiles(StepDir), Result);
  }

  return Result;
}

void Runner::resetDirtyness() {
  for (auto &Step : Steps) {
    for (auto &Container : Step.second.containers()) {
      if (Container.second == nullptr)
        continue;
      Container.second->resetDirtiness();
    }
  }
}

llvm::Expected<DiffMap>
Runner::runAnalysis(llvm::StringRef AnalysisName,
                    llvm::StringRef StepName,
                    const ContainerToTargetsMap &Targets,
                    TargetInStepSet &InvalidationsMap,
                    const llvm::StringMap<std::string> &Options) {

  GlobalsMap Before = getContext().getGlobals();

  auto MaybeStep = Steps.find(StepName);

  if (MaybeStep == Steps.end()) {
    return revng::createError("Could not find a step named %s\n",
                              StepName.str().c_str());
  }

  Task T(3, "Analysis execution");
  T.advance("Produce step " + StepName, true);
  if (llvm::Error Error = run(StepName, Targets))
    return std::move(Error);

  T.advance("Run analysis", true);
  if (llvm::Error Error = MaybeStep->second.runAnalysis(AnalysisName,
                                                        Targets,
                                                        Options);
      Error) {
    return std::move(Error);
  }

  T.advance("Apply diff produced by the analysis", true);
  const GlobalsMap &After = getContext().getGlobals();
  DiffMap Map = Before.diff(After);
  for (const auto &GlobalNameDiffPair : Map)
    if (llvm::Error Error = apply(GlobalNameDiffPair.second, InvalidationsMap))
      return std::move(Error);

  return std::move(Map);
}

Error Runner::run(const State &ToProduce) {
  Task T(ToProduce.size(), "Multi-step pipeline run");
  for (const auto &Request : ToProduce) {
    T.advance(Request.first(), true);
    if (llvm::Error Error = run(Request.first(), Request.second))
      return Error;
  }

  return llvm::Error::success();
}

Error Runner::run(llvm::StringRef EndingStepName,
                  const ContainerToTargetsMap &Targets) {

  vector<PipelineExecutionEntry> ToExec;
  revng_log(ExplanationLogger, "Running until step " << EndingStepName);

  if (llvm::Error Error = getObjectives(*this, EndingStepName, Targets, ToExec);
      Error) {
    return Error;
  }

  explainPipeline(Targets, ToExec);

  if (ToExec.size() <= 1)
    return Error::success();

  for (PipelineExecutionEntry &StepGoalsPairs : llvm::drop_begin(ToExec)) {
    Step &Step = *StepGoalsPairs.ToExecute;
    if (llvm::Error Error = Step.checkPrecondition()) {
      return llvm::make_error<AnnotatedError>(std::move(Error),
                                              "While scheduling step "
                                                + Step.getName() + ":");
    }
  }

  Task T(ToExec.size() - 1, "Produce steps required up to " + EndingStepName);
  for (PipelineExecutionEntry &StepGoalsPairs : llvm::drop_begin(ToExec)) {
    auto &[Step, PredictedOutput, Input, PipesInfo] = StepGoalsPairs;
    T.advance(Step->getName(), true);

    Task T2(3, "Run step");
    T2.advance("Clone and filter input containers", true);

    ::Step &Parent = Step->getPredecessor();
    ContainerSet CurrentContainer = Parent.containers().cloneFiltered(Input);

    // Run the step
    T2.advance("Run the step", true);
    Step->run(std::move(CurrentContainer), PipesInfo);

    T2.advance("Extract the requested targets", true);
    if (VerifyLog.isEnabled()) {
      ContainerSet Produced = Step->containers().cloneFiltered(PredictedOutput);

      if (not Produced.enumerate().contains(PredictedOutput)) {
        dbg << "PredictedOutput:\n";
        PredictedOutput.dump(dbg, 2, false);
        dbg << "Produced:\n";
        Produced.enumerate().dump(dbg, 2, false);
        revng_abort("Not all the expected targets have been produced");
      }
      revng_check(Step->containers().enumerate().contains(PredictedOutput));
    }
  }

  if (ExplanationLogger.isEnabled()) {
    ExplanationLogger << "PRODUCED\n";
    indent(ExplanationLogger, 1);
    ExplanationLogger << EndingStepName << ":\n";
    ToExec.back().ToExecute->containers().enumerate().dump(ExplanationLogger,
                                                           2,
                                                           false);
    ExplanationLogger << DoLog;
  }

  return Error::success();
}

Error Runner::invalidate(const TargetInStepSet &Invalidations) {
  for (const auto &Step : Invalidations) {
    llvm::StringRef StepName = Step.first();
    const ContainerToTargetsMap &ToRemove = Step.second;
    if (llvm::Error Error = operator[](StepName).invalidate(ToRemove))
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
    auto Result = NextStep.deduceResults(Out[Step.getName()]);
    Out[NextStep.getName()].merge(std::move(Result));
  }
}
const KindsRegistry &Runner::getKindsRegistry() const {
  return TheContext->getKindsRegistry();
}

void Runner::getDiffInvalidations(const GlobalTupleTreeDiff &Diff,
                                  TargetInStepSet &Map) const {

  // Iterate over each step, skipping the begin step
  for (const Step &Step : llvm::drop_begin(*this)) {
    revng_log(InvalidationLog,
              "Computing invalidations in step " << Step.getName());
    LoggerIndent<> I(InvalidationLog);

    ContainerToTargetsMap &StepInvalidations = Map[Step.getName()];

    // Iterate over each pipe and feed it the non-const containers
    // TODO: this is misguided! The pipe are going to be looking at the
    //       containers *at the end* of the pipe. This might mean that what
    //       we're looking for might no longer be there!
    Step.pipeInvalidate(Diff, StepInvalidations);

    // Compute the set of things we need to invalidate by looking at all the
    // paths changed by Diff.
    // Also, this will invalidate all the targets depending on them and all the
    // targets depending on stuff that's already in Map.
    revng_log(InvalidationLog, Diff.getPaths().size() << " paths have changed");
    for (const TupleTreePath *Path : Diff.getPaths()) {
      revng_log(InvalidationLog, "Processing " << *Diff.pathAsString(*Path));
      LoggerIndent<> Indent(InvalidationLog);
      Step.registerTargetsDependingOn(Diff.getGlobalName(),
                                      *Path,
                                      Map,
                                      InvalidationLog);
    }
  }
}

llvm::Error Runner::apply(const GlobalTupleTreeDiff &Diff,
                          TargetInStepSet &Map) {
  getDiffInvalidations(Diff, Map);
  if (auto Error = getInvalidations(Map))
    return Error;

  return invalidate(Map);
}
