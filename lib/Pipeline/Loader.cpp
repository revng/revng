/// \file Loader.cpp
/// \brief a loader is a object that acceps a serialized pipeline and yields a
/// runner object that rappresents that pipeline.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/StringRef.h"

#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Runner.h"

using namespace pipeline;
using namespace std;
using namespace llvm;

llvm::Error Loader::parseStepDeclaration(Runner &Runner,
                                         const StepDeclaration &Declaration,
                                         std::string &LastAddedStep) const {
  auto &JustAdded = Runner.emplaceStep(LastAddedStep, Declaration.Name);
  LastAddedStep = Declaration.Name;
  for (const auto &Invocation : Declaration.Pipes) {
    if (not isInvocationUsed(Invocation.EnabledWhen))
      continue;

    if (auto Error = parseInvocation(JustAdded, Invocation); !!Error)
      return Error;
  }

  return Error::success();
}

llvm::Expected<std::unique_ptr<LLVMPassWrapperBase>>
Loader::loadPassFromName(llvm::StringRef Name) const {

  auto It = KnownLLVMPipeTypes.find(Name);
  if (It != KnownLLVMPipeTypes.end())
    return It->second();

  return PureLLVMPassWrapper::create(Name);
}

llvm::Error
Loader::parseLLVMPass(Step &Step, const PipeInvocation &Invocation) const {

  LLVMPipe ToInsert;

  if (OnLLVMContainerCreationAction.has_value()) {
    auto MaybeError = (*OnLLVMContainerCreationAction)(*this, ToInsert);
    if (!!MaybeError)
      return MaybeError;
  }

  for (const auto &PassName : Invocation.Passes) {
    auto MaybePass = loadPassFromName(PassName);
    if (not MaybePass)
      return MaybePass.takeError();
    ToInsert.addPass(std::move(*MaybePass));
  }

  auto Wrapper = PipeWrapper(move(ToInsert), Invocation.UsedContainers);
  Step.addPipe(move(Wrapper));

  return Error::success();
}

llvm::Error
Loader::parseInvocation(Step &Step, const PipeInvocation &Invocation) const {
  if (Invocation.Name == "LLVMPipe")
    return parseLLVMPass(Step, Invocation);

  auto It = KnownPipesTypes.find(Invocation.Name);
  if (It == KnownPipesTypes.end()) {
    auto *Message = "while parsing pipe invocation: No known Pipe with "
                    "name %s ";
    return createStringError(inconvertibleErrorCode(),
                             Message,
                             Invocation.Name.c_str());
  }
  auto &Entry = It->second;
  Step.addPipe(Entry(Invocation.UsedContainers));
  return Error::success();
}

using BCDecl = ContainerDeclaration;
Error Loader::parseContainerDeclaration(Runner &Pipeline,
                                        const BCDecl &Dec) const {
  auto It = KnownContainerTypes.find(Dec.Type);
  if (It == KnownContainerTypes.end()) {
    auto *Message = "while parsing container declaration: No known container "
                    "with name %s";
    return createStringError(inconvertibleErrorCode(),
                             Message,
                             Dec.Type.c_str());
  }
  auto &Entry = It->second;
  Pipeline.addContainerFactory(Dec.Name, Entry);

  return Error::success();
}

llvm::Expected<Runner>
Loader::load(llvm::ArrayRef<std::string> Pipelines) const {
  std::vector<PipelineDeclaration> Declarations(Pipelines.size());
  for (size_t I = 0; I < Pipelines.size(); I++) {
    yaml::Input Input(Pipelines[I]);

    Input >> Declarations[I];
    if (Input.error())
      return createStringError(inconvertibleErrorCode(),
                               "Could not parse pipeline");
  }

  return load(Declarations);
}
void Loader::emitTerminators(Runner &Runner) const {
  auto LeafsCount = llvm::count_if(Runner, [&Runner](const Step &CurrentStep) {
    return not Runner.hasSuccessors(CurrentStep);
  });

  for (const Step &CurrentStep : Runner)
    if (not Runner.hasSuccessors(CurrentStep)) {
      std::string
        Name = (LeafsCount == 1 ? "End" : "End" + CurrentStep.getName()).str();
      Runner.emplaceStep(CurrentStep.getName().str(), std::move(Name));
    }
}

llvm::Error Loader::parseSteps(Runner &Runner,
                               const PipelineDeclaration &Declaration) const {

  std::string LastAddedStep = Declaration.From;
  for (const auto &Step : Declaration.Steps) {
    if (not isInvocationUsed(Step.EnabledWhen))
      continue;

    if (auto Error = parseStepDeclaration(Runner, Step, LastAddedStep); !!Error)
      return Error;
  }
  return llvm::Error::success();
}

llvm::Error
Loader::parseDeclarations(Runner &Runner,
                          const PipelineDeclaration &Declaration) const {
  for (const auto &Container : Declaration.Containers)
    if (auto Error = parseContainerDeclaration(Runner, Container); !!Error)
      return Error;
  return llvm::Error::success();
}

static bool
requirementsMet(llvm::ArrayRef<const PipelineDeclaration *> Scheduled,
                const PipelineDeclaration &ToSchedule) {
  if (ToSchedule.From.empty())
    return true;

  for (const auto *Declaration : Scheduled)
    for (const auto &StepName : Declaration->Steps)
      if (StepName.Name == ToSchedule.From)
        return true;

  return false;
}

static llvm::Error
sortPipeline(llvm::SmallVector<const PipelineDeclaration *, 2> &ToSort) {
  llvm::SmallVector<const PipelineDeclaration *> Sorted;

  const auto RequirementsMet = [&](const PipelineDeclaration *Declaration) {
    if (not requirementsMet(Sorted, *Declaration))
      return false;

    Sorted.push_back(Declaration);
    return true;
  };

  while (not ToSort.empty()) {
    size_t InitialSize = ToSort.size();
    llvm::erase_if(ToSort, RequirementsMet);
    if (InitialSize == ToSort.size())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Could not satisfy all requirements");
  }

  ToSort = std::move(Sorted);
  return llvm::Error::success();
}

llvm::Expected<Runner>
Loader::load(llvm::ArrayRef<PipelineDeclaration> Pipelines) const {
  Runner ToReturn(*PipelineContext);

  llvm::SmallVector<const PipelineDeclaration *, 2> ToSort;
  for (const auto &Pipeline : Pipelines)
    ToSort.push_back(&Pipeline);

  if (auto Error = sortPipeline(ToSort); Error)
    return std::move(Error);

  for (const auto *Declaration : ToSort)
    if (auto Error = parseDeclarations(ToReturn, *Declaration); Error)
      return std::move(Error);

  for (const auto *Declaration : ToSort)
    if (auto Error = parseSteps(ToReturn, *Declaration); Error)
      return std::move(Error);
  emitTerminators(ToReturn);

  return ToReturn;
}

llvm::Expected<Runner>
Loader::load(const PipelineDeclaration &Declaration) const {
  ArrayRef<PipelineDeclaration> Pipelines(&Declaration, 1);
  return load(Pipelines);
}

bool Loader::isInvocationUsed(const vector<string> &Invocation) const {
  if (Invocation.size() == 0)
    return true;

  const auto IsStringEnabled = [this](const std::string &Name) {
    if (not Name.starts_with("~"))
      return EnabledFlags.count(Name) != 0;

    string ActualName(Name.begin() + 1, Name.end());
    return EnabledFlags.count(ActualName) == 0;
  };
  return llvm::any_of(Invocation, IsStringEnabled);
}
