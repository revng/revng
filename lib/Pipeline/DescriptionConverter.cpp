/// \file DescriptionConverter.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/AnalysesList.h"
#include "revng/Pipeline/Analysis.h"
#include "revng/Pipeline/Description/PipelineDescription.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Step.h"

static pipeline::description::Analysis
describe(const pipeline::AnalysisWrapper &Analysis) {
  pipeline::description::Analysis Result;

  Result.Name() = Analysis->getUserBoundName();
  Result.Available() = Analysis->isAvailable();

  auto ContainerInputs = Analysis->getRunningContainersNames();
  for (size_t I = 0; I < ContainerInputs.size(); I++) {
    pipeline::description::AnalysisContainerInput Input;
    Input.Name() = ContainerInputs[I];
    for (auto &Kind : Analysis->getAcceptedKinds(I))
      Input.AcceptableKinds().insert(Kind->name().str());

    Result.ContainerInputs().insert(Input);
  }

  std::vector<std::string> OptionNames = Analysis->getOptionsNames();
  std::vector<std::string> OptionTypes = Analysis->getOptionsTypes();
  for (size_t I = 0; I < OptionNames.size(); I++)
    Result.Options().insert({ OptionNames[I], OptionTypes[I] });

  return Result;
}

static pipeline::description::AnalysesList
describe(const pipeline::AnalysesList &List) {
  pipeline::description::AnalysesList Result;

  Result.Name() = List.getName();

  for (auto &Analysis : List) {
    Result.Analyses().insert({ Analysis.getAnalysisName().str(),
                               Analysis.getStepName().str() });
  }

  return Result;
}

static pipeline::description::Kind describe(const pipeline::Kind &Kind) {
  pipeline::description::Kind Result;

  Result.Name() = Kind.name();
  Result.Rank() = Kind.rank().name();
  if (const pipeline::Kind *Parent = Kind.parent(); Parent != nullptr)
    Result.Parent() = Parent->name();

  for (const pipeline::Rank *Rank : Kind.definedLocations())
    Result.DefinedLocations().insert(Rank->name().str());

  for (const pipeline::Kind *PrefKind : Kind.preferredKinds())
    Result.PreferredKinds().insert(PrefKind->name().str());

  return Result;
}

static pipeline::description::Rank describe(const pipeline::Rank &Rank) {
  pipeline::description::Rank Result;

  Result.Name() = Rank.name();
  if (const pipeline::Rank *Parent = Rank.parent(); Parent != nullptr)
    Result.Parent() = Parent->name();

  Result.Depth() = Rank.depth();

  return Result;
}

static pipeline::description::Step describe(const pipeline::Step &Step) {
  pipeline::description::Step Result;

  Result.Name() = Step.getName();
  Result.Component() = Step.getComponent().str();
  if (Step.hasPredecessor())
    Result.Parent() = Step.getPredecessor().getName();

  for (auto &[Name, RunnerAnalysis] : Step.analyses())
    Result.Analyses().insert(describe(RunnerAnalysis));

  if (const pipeline::Kind *Kind = Step.getArtifactsKind(); Kind != nullptr) {
    pipeline::description::Artifacts Artifacts;
    Artifacts.Kind() = Kind->name();
    Artifacts.Container() = Step.getArtifactsContainerName();
    Artifacts.SingleTargetFilename() = Step.getArtifactsSingleTargetFilename();
    Result.Artifacts() = Artifacts;
  }

  return Result;
}

pipeline::description::PipelineDescription
pipeline::Runner::description() const {
  pipeline::description::PipelineDescription Result;

  for (size_t I = 0; I < getAnalysesListCount(); I++) {
    const pipeline::AnalysesList &List = getAnalysesList(I);
    Result.AnalysesLists().insert(describe(List));
  }

  const pipeline::GlobalsMap &GlobalsMap = getContext().getGlobals();
  for (size_t I = 0; I < GlobalsMap.size(); I++)
    Result.Globals().insert(GlobalsMap.getName(I).str());

  for (pipeline::Kind &Kind : getKindsRegistry())
    Result.Kinds().insert(describe(Kind));

  for (pipeline::Rank *Rank : pipeline::Rank::getAll())
    Result.Ranks().insert(describe(*Rank));

  const auto &ContainerRegistry = getContainerFactorySet();
  for (const auto &[Name, ContainerIdentifier] : ContainerRegistry) {
    Result.Containers().insert({ Name.str(),
                                 ContainerIdentifier.mimeType().str() });
  }

  for (auto &Step : *this)
    Result.Steps().insert(describe(Step));

  return Result;
}
