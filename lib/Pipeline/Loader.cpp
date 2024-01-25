/// \file Loader.cpp
/// A loader is a object that acceps a serialized pipeline and yields a runner
/// object that rappresents that pipeline.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "revng/Pipeline/Loader.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Pipeline/Runner.h"

using namespace pipeline;
using namespace std;
using namespace llvm;
using StringsMap = llvm::StringMap<string>;

Error Loader::parseStepDeclaration(Runner &Runner,
                                   const StepDeclaration &Declaration,
                                   std::string &LastAddedStep,
                                   const StringsMap &ReadOnlyNames,
                                   const llvm::StringRef Component) const {
  auto &JustAdded = Runner.emplaceStep(LastAddedStep,
                                       Declaration.Name,
                                       Component);
  LastAddedStep = Declaration.Name;

  if (Declaration.Artifacts.isValid()) {
    auto &KindName = Declaration.Artifacts.Kind;
    const Kind *Kind = Runner.getKindsRegistry().find(KindName);
    if (Kind == nullptr) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     Twine("Can't find kind ") + KindName
                                       + Twine(" for the artifact of step ")
                                       + LastAddedStep + Twine(" not found"));
    }
    if (auto Error = JustAdded.setArtifacts(Declaration.Artifacts.Container,
                                            Kind,
                                            Declaration.Artifacts
                                              .SingleTargetFilename);
        !!Error) {
      return Error;
    }
  }

  for (const auto &Invocation : Declaration.Pipes) {
    if (not isInvocationUsed(Invocation.EnabledWhen))
      continue;

    if (auto MaybeInvocation = parseInvocation(JustAdded,
                                               Invocation,
                                               ReadOnlyNames);
        !MaybeInvocation)
      return MaybeInvocation.takeError();
    else
      JustAdded.addPipe(std::move(*MaybeInvocation));
  }

  for (const auto &SingleAnalysis : Declaration.Analyses) {
    auto MaybeInvocation = parseAnalysis(SingleAnalysis);
    if (!MaybeInvocation)
      return MaybeInvocation.takeError();

    JustAdded.addAnalysis(SingleAnalysis.Name, std::move(*MaybeInvocation));
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

llvm::Expected<PipeWrapper>
Loader::parseLLVMPass(const PipeInvocation &Invocation) const {

  LLVMPipe ToInsert;

  if (OnLLVMContainerCreationAction.has_value()) {
    auto MaybeError = (*OnLLVMContainerCreationAction)(*this, ToInsert);
    if (!!MaybeError)
      return std::move(MaybeError);
  }

  for (const auto &PassName : Invocation.Passes) {
    auto MaybePass = loadPassFromName(PassName);
    if (not MaybePass)
      return MaybePass.takeError();
    ToInsert.addPass(std::move(*MaybePass));
  }

  return PipeWrapper::make(std::move(ToInsert), Invocation.UsedContainers);
}

llvm::Expected<AnalysisWrapper>
Loader::parseAnalysis(const AnalysisDeclaration &Declaration) const {
  auto It = KnownAnalysisTypes.find(Declaration.Type);
  if (It == KnownAnalysisTypes.end()) {
    auto *Message = "While parsing analyses: no known analysis named name %s\n";
    return createStringError(inconvertibleErrorCode(),
                             Message,
                             Declaration.Type.c_str());
  }
  auto &Entry = It->second;
  auto ToReturn = AnalysisWrapper(Entry, Declaration.UsedContainers);
  ToReturn->setUserBoundName(Declaration.Name);
  return ToReturn;
}

llvm::Expected<PipeWrapper>
Loader::parseInvocation(Step &Step,
                        const PipeInvocation &Invocation,
                        const StringsMap &ReadOnlyNames) const {
  if (Invocation.Type == "LLVMPipe")
    return parseLLVMPass(Invocation);

  if (not Invocation.Passes.empty())
    return createStringError(inconvertibleErrorCode(),
                             "while parsing pipe %s: Passes declarations are "
                             "not allowed in non-llvm pipes\n",
                             Invocation.Type.c_str());

  auto It = KnownPipesTypes.find(Invocation.Type);
  if (It == KnownPipesTypes.end()) {
    auto *Message = "While parsing pipe invocation: no known pipe with "
                    "name %s\n";
    return createStringError(inconvertibleErrorCode(),
                             Message,
                             Invocation.Type.c_str());
  }
  auto &Pipe = It->second;
  for (const auto &ContainerNameAndIndex :
       llvm::enumerate(Invocation.UsedContainers)) {

    const auto &ContainerName = ContainerNameAndIndex.value();
    size_t Index = ContainerNameAndIndex.index();
    if (ReadOnlyNames.find(ContainerName) == ReadOnlyNames.end())
      continue;

    const auto &RoleName = ReadOnlyNames.find(ContainerName)->second;
    if (Pipe.Pipe->isContainerArgumentConst(Index))
      continue;

    if (PipelineContext->hasRegisteredReadOnlyContainer(ContainerName)) {
      return createStringError(inconvertibleErrorCode(),
                               "Detected two non const uses of read only "
                               "container %s\n",
                               ContainerName.c_str());
    }

    revng_assert(Step.containers().containsOrCanCreate(ContainerName));
    const auto &Container = *Step.containers().find(ContainerName);
    PipelineContext->addReadOnlyContainer(RoleName, Container);
  }

  return PipeWrapper(Pipe, Invocation.UsedContainers);
}

using BCDecl = ContainerDeclaration;
Error Loader::parseContainerDeclaration(Runner &Pipeline,
                                        const BCDecl &Dec,
                                        StringsMap &ReadOnlyNames) const {
  if (not Dec.Role.empty() and KnownContainerRoles.count(Dec.Role) == 0) {
    auto *Message = "While parsing container declaration with Name %s has a "
                    "unknown role %s.\n";
    return createStringError(inconvertibleErrorCode(),
                             Message,
                             Dec.Name.c_str(),
                             Dec.Role.c_str());
  }

  if (not Dec.Role.empty()
      and KnownContainerRoles.find(Dec.Role)->second != Dec.Type) {
    auto *Message = "While parsing container declaration with Name %s: "
                    "role %s was not a valid role for container of type %s.\n";
    return createStringError(inconvertibleErrorCode(),
                             Message,
                             Dec.Name.c_str(),
                             Dec.Role.c_str(),
                             Dec.Type.c_str());
  }

  auto It = KnownContainerTypes.find(Dec.Type);
  if (It == KnownContainerTypes.end()) {
    auto *Message = "While parsing container declaration: No known container "
                    "with name %s\n";
    return createStringError(inconvertibleErrorCode(),
                             Message,
                             Dec.Type.c_str());
  }
  auto &Entry = It->second;
  Pipeline.addContainerFactory(Dec.Name, Entry);
  if (not Dec.Role.empty())
    ReadOnlyNames[Dec.Name] = Dec.Role;

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
                               "Could not parse pipeline\n");
  }

  return load(Declarations);
}

llvm::Error Loader::parseSteps(Runner &Runner,
                               const BranchDeclaration &Declaration,
                               const StringsMap &ReadOnlyNames,
                               const llvm::StringRef Component) const {

  std::string LastAddedStep = Declaration.From.empty() ? "begin" :
                                                         Declaration.From;
  for (const auto &Step : Declaration.Steps) {
    if (not isInvocationUsed(Step.EnabledWhen))
      continue;

    if (auto Error = parseStepDeclaration(Runner,
                                          Step,
                                          LastAddedStep,
                                          ReadOnlyNames,
                                          Component);
        !!Error)
      return Error;
  }
  return llvm::Error::success();
}

llvm::Error Loader::parseDeclarations(Runner &Runner,
                                      const PipelineDeclaration &Declaration,
                                      StringsMap &ReadOnlyNames) const {

  for (const auto &Container : Declaration.Containers)
    if (auto Error = parseContainerDeclaration(Runner,
                                               Container,
                                               ReadOnlyNames);
        !!Error)
      return Error;
  return llvm::Error::success();
}

static bool requirementsMet(llvm::ArrayRef<const BranchDeclaration *> Scheduled,
                            const BranchDeclaration &ToSchedule) {
  if (ToSchedule.From.empty())
    return true;

  for (const auto *Declaration : Scheduled)
    for (const auto &StepName : Declaration->Steps)
      if (StepName.Name == ToSchedule.From)
        return true;

  return false;
}

static llvm::Error
sortPipeline(llvm::SmallVector<const BranchDeclaration *, 2> &ToSort) {
  llvm::SmallVector<const BranchDeclaration *> Sorted;

  const auto RequirementsMet = [&](const BranchDeclaration *Declaration) {
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

  llvm::SmallVector<const BranchDeclaration *, 2> ToSort;
  std::map<const BranchDeclaration *, llvm::StringRef> BranchToComponent;
  for (const auto &Pipeline : Pipelines) {
    for (const auto &Declaration : Pipeline.Branches) {
      ToSort.push_back(&Declaration);
      BranchToComponent[&Declaration] = Pipeline.Component;
    }
  }

  if (auto Error = sortPipeline(ToSort); Error)
    return std::move(Error);

  llvm::StringMap<std::string> ReadOnlyNames;
  for (const auto &Declaration : Pipelines)
    if (auto Error = parseDeclarations(ToReturn, Declaration, ReadOnlyNames);
        Error)
      return std::move(Error);

  ToReturn.emplaceStep("", "begin", "");

  for (const auto *Declaration : ToSort) {
    llvm::StringRef Component = BranchToComponent[Declaration];
    if (auto Error = parseSteps(ToReturn,
                                *Declaration,
                                ReadOnlyNames,
                                Component);
        Error)
      return std::move(Error);
  }

  for (const auto &Declaration : Pipelines)
    for (const auto &Analysis : Declaration.Analyses) {
      auto MaybeAnalysis = parseAnalysis(Analysis);
      if (not MaybeAnalysis)
        return MaybeAnalysis.takeError();
      ToReturn.getStep(Analysis.Step)
        .addAnalysis(Analysis.Name, std::move(*MaybeAnalysis));
    }

  for (const auto &Declaration : Pipelines)
    for (const auto &List : Declaration.AnalysesLists) {
      llvm::SmallVector<AnalysisReference, 2> Analyses;
      for (auto &AnalysisName : List.UsedAnalysesNames) {
        if (not ToReturn.containsAnalysis(AnalysisName)) {
          return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                         "No known analysis " + AnalysisName
                                           + " used in analysis list "
                                           + List.Name);
        }

        for (const auto &Step : ToReturn) {
          if (Step.hasAnalysis(AnalysisName)) {
            Analyses.emplace_back(Step.getName().str(), AnalysisName);
            break;
          }
        }
      }

      ToReturn.addAnalysesList(List.Name, Analyses);
    }

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
      return EnabledFlags.contains(Name);

    string ActualName(Name.begin() + 1, Name.end());
    return !EnabledFlags.contains(ActualName);
  };
  return llvm::any_of(Invocation, IsStringEnabled);
}
