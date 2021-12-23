/// \file Runner.cpp
/// \brief a step is composed of a list of pipes and a set of containers
/// rappresenting the content of the pipeline before the execution of such
/// pipes.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include "revng/Pipeline/Step.h"
#include "revng/Support/Debug.h"

using namespace llvm;
using namespace std;
using namespace pipeline;

PipeWrapper &PipeWrapper::operator=(const PipeWrapper &Other) {
  if (this == &Other)
    return *this;

  Pipe = Other.Pipe->clone();
  return *this;
}

ContainerToTargetsMap
Step::analyzeGoals(const ContainerToTargetsMap &RequiredGoals,
                   ContainerToTargetsMap &AlreadyAviable) const {

  ContainerToTargetsMap Targets = RequiredGoals;
  for (const auto &Pipe : llvm::make_range(Pipes.rbegin(), Pipes.rend())) {
    Targets = Pipe->getRequirements(Targets);
  }
  removeSatisfiedGoals(Targets, AlreadyAviable);

  return Targets;
}

void Step::explainStartStep(const ContainerToTargetsMap &Targets,
                            llvm::raw_ostream *OS,
                            size_t Indentation) const {
  if (OS == nullptr)
    return;

  (*OS) << "\n";
  OS->changeColor(llvm::raw_ostream::Colors::GREEN);
  (*OS) << "Starting Step: ";
  OS->changeColor(llvm::raw_ostream::Colors::MAGENTA);
  (*OS) << getName();
  OS->changeColor(llvm::raw_ostream::Colors::GREEN);
  (*OS) << " by cloning\n";

  prettyPrintStatus(Targets, *OS, Indentation + 1);
}

void Step::explainExecutedPipe(const PipeWrapper &Wrapper,
                               llvm::raw_ostream *OS,
                               size_t Indentation) const {
  if (OS == nullptr)
    return;

  OS->changeColor(llvm::raw_ostream::Colors::GREEN);
  (*OS) << Wrapper->getName();
  OS->changeColor(llvm::raw_ostream::Colors::GREEN);
  (*OS) << "(";

  auto Vec = Wrapper->getRunningContainersNames();
  if (not Vec.empty()) {
    for (size_t I = 0; I < Vec.size() - 1; I++) {
      OS->changeColor(llvm::raw_ostream::Colors::BLUE);
      (*OS) << Vec[I];
      OS->changeColor(llvm::raw_ostream::Colors::GREEN);
      (*OS) << ", ";
    }
    OS->changeColor(llvm::raw_ostream::Colors::BLUE);
    (*OS) << Vec.back();
  }

  OS->changeColor(llvm::raw_ostream::Colors::GREEN);
  (*OS) << ")";
  (*OS) << "\n";
}

ContainerSet Step::cloneAndRun(Context &Ctx,
                               const ContainerToTargetsMap &Targets,
                               llvm::raw_ostream *OS) {
  auto RunningContainers = Containers.cloneFiltered(Targets);
  explainStartStep(Targets, OS);

  for (auto &Pipe : Pipes) {
    if (not Pipe->areRequirementsMet(RunningContainers.enumerate()))
      continue;

    explainExecutedPipe(Pipe, OS);
    Pipe->run(Ctx, RunningContainers);
  }
  return RunningContainers;
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

ContainerToTargetsMap Step::deduceResults(ContainerToTargetsMap Input) const {
  for (const auto &Pipe : Pipes)
    Input = Pipe->deduceResults(Input);
  return Input;
}

Error Step::invalidate(const ContainerToTargetsMap &ToRemove) {
  return containers().remove(ToRemove);
}

Error Step::storeToDisk(llvm::StringRef DirPath) const {
  auto Path = DirPath.str() + "/" + Name;
  if (auto ErrorCode = llvm::sys::fs::create_directories(Path); ErrorCode)
    return createStringError(ErrorCode,
                             "Could not create dir %s",
                             Path.c_str());
  return Containers.storeToDisk(Path);
}
Error Step::loadFromDisk(llvm::StringRef DirPath) {
  auto Path = DirPath.str() + "/" + Name;
  if (not llvm::sys::fs::exists(Path))
    return llvm::Error::success();
  return Containers.loadFromDisk(Path);
}
