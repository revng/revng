/// \file Runner.cpp
/// A step is composed of a list of pipes and a set of containers representing
/// the content of the pipeline before the execution of such pipes.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Progress.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Errors.h"
#include "revng/Pipeline/Step.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

using namespace llvm;
using namespace std;
using namespace pipeline;

ContainerToTargetsMap
Step::analyzeGoals(const Context &Ctx,
                   const ContainerToTargetsMap &RequiredGoals) const {

  ContainerToTargetsMap AlreadyAvailable;
  ContainerToTargetsMap Targets = RequiredGoals;
  removeSatisfiedGoals(Targets, AlreadyAvailable);
  for (const auto &Pipe : llvm::make_range(Pipes.rbegin(), Pipes.rend())) {
    Targets = Pipe->getRequirements(Ctx, Targets);
  }

  return Targets;
}

void Step::explainStartStep(const ContainerToTargetsMap &Targets,
                            size_t Indentation) const {

  indent(ExplanationLogger, Indentation);
  ExplanationLogger << "STARTING step on containers\n";
  indent(ExplanationLogger, Indentation + 1);
  ExplanationLogger << getName() << ":\n";
  prettyPrintStatus(Targets, ExplanationLogger, Indentation + 2);
  ExplanationLogger << DoLog;
}

void Step::explainEndStep(const ContainerToTargetsMap &Targets,
                          size_t Indentation) const {

  indent(ExplanationLogger, Indentation);
  ExplanationLogger << "ENDING step, the following have been produced\n";
  indent(ExplanationLogger, Indentation + 1);
  ExplanationLogger << getName() << ":\n";
  prettyPrintStatus(Targets, ExplanationLogger, Indentation + 2);
  ExplanationLogger << DoLog;
}

void Step::explainExecutedPipe(const Context &Ctx,
                               const InvokableWrapperBase &Wrapper,
                               size_t Indentation) const {
  ExplanationLogger << "RUN " << Wrapper.getName();
  ExplanationLogger << "(";

  auto Vec = Wrapper.getRunningContainersNames();
  if (not Vec.empty()) {
    for (size_t I = 0; I < Vec.size() - 1; I++) {
      ExplanationLogger << Vec[I];
      ExplanationLogger << ", ";
    }
    ExplanationLogger << Vec.back();
  }

  ExplanationLogger << ")";
  ExplanationLogger << "\n";
  ExplanationLogger << DoLog;

  auto CommandStream = CommandLogger.getAsLLVMStream();
  Wrapper.print(Ctx, *CommandStream, Indentation);
  CommandStream->flush();
  CommandLogger << DoLog;
}

ContainerSet Step::run(Context &Ctx, ContainerSet &&Input) {
  auto InputEnumeration = Input.enumerate();
  explainStartStep(InputEnumeration);

  Task T(Pipes.size() + 1, "Step " + getName());
  for (PipeWrapper &Pipe : Pipes) {
    T.advance(Pipe->getName(), false);
    explainExecutedPipe(Ctx, *Pipe);
    cantFail(Pipe->run(Ctx, Input));
    llvm::cantFail(Input.verify());
  }

  T.advance("Merging back", true);
  explainEndStep(Input.enumerate());
  Containers.mergeBack(std::move(Input));
  InputEnumeration = deduceResults(Ctx, InputEnumeration);
  auto Cloned = Containers.cloneFiltered(InputEnumeration);
  return Cloned;
}

llvm::Error Step::runAnalysis(llvm::StringRef AnalysisName,
                              Context &Ctx,
                              const ContainerToTargetsMap &Targets,
                              const llvm::StringMap<std::string> &ExtraArgs) {
  auto Stream = ExplanationLogger.getAsLLVMStream();
  ContainerToTargetsMap Map = Containers.enumerate();

  auto CollapsedTargets = Targets;

  revng_assert(Map.contains(CollapsedTargets),
               "An analysis was requested, but not all targets are available");

  auto &TheAnalysis = getAnalysis(AnalysisName);

  explainExecutedPipe(Ctx, *TheAnalysis);

  auto Cloned = Containers.cloneFiltered(Targets);
  return TheAnalysis->run(Ctx, Cloned, ExtraArgs);
}

void Step::removeSatisfiedGoals(TargetsList &RequiredInputs,
                                const ContainerBase &CachedSymbols,
                                TargetsList &ToLoad) {
  const TargetsList EnumeratedSymbols = CachedSymbols.enumerate();
  const auto IsCached = [&ToLoad,
                         &EnumeratedSymbols](const Target &Target) -> bool {
    bool MustBeLoaded = EnumeratedSymbols.contains(Target);
    if (MustBeLoaded)
      ToLoad.emplace_back(Target);
    return MustBeLoaded;
  };

  llvm::erase_if(RequiredInputs, IsCached);
}

void Step::removeSatisfiedGoals(ContainerToTargetsMap &Targets,
                                ContainerToTargetsMap &ToLoad) const {
  for (auto &RequiredInputsFromContainer : Targets) {
    llvm::StringRef ContainerName = RequiredInputsFromContainer.first();
    auto &RequiredInputs = RequiredInputsFromContainer.second;
    auto &ToLoadFromCurrentContainer = ToLoad[ContainerName];
    if (Containers.contains(ContainerName))
      removeSatisfiedGoals(RequiredInputs,
                           Containers.at(ContainerName),
                           ToLoadFromCurrentContainer);
  }
}

ContainerToTargetsMap Step::deduceResults(const Context &Ctx,
                                          ContainerToTargetsMap Input) const {
  for (const auto &Pipe : Pipes)
    Input = Pipe->deduceResults(Ctx, Input);
  return Input;
}

Error Step::invalidate(const ContainerToTargetsMap &ToRemove) {
  return containers().remove(ToRemove);
}

Error Step::store(const revng::DirectoryPath &DirPath) const {
  return Containers.store(DirPath);
}

Error Step::checkPrecondition(const Context &Ctx) const {
  for (const auto &Pipe : Pipes) {
    if (auto Error = Pipe->checkPrecondition(Ctx); Error)
      return llvm::make_error<AnnotatedError>(std::move(Error),
                                              "While scheduling pipe "
                                                + Pipe->getName() + ":");
  }
  return llvm::Error::success();
}

Error Step::load(const revng::DirectoryPath &DirPath) {
  return Containers.load(DirPath);
}
