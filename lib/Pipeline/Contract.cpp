/// \file Contract.cpp
/// \brief A contract is "Rule" attached to a pipe that specifies what kind of
/// transformations the pipe is allowed to do on input containers.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Target.h"

using namespace pipeline;
using namespace llvm;
using namespace std;

void Contract::deduceResults(ContainerToTargetsMap &StepStatus,
                             ArrayRef<string> Names) const {
  auto &OutputContainerTarget = StepStatus[Names[PipeArgumentTargetIndex]];

  TargetsList Tmp;
  deduceResults(StepStatus, Tmp, Names);

  copy(Tmp, back_inserter(OutputContainerTarget));
}

void Contract::deduceResults(ContainerToTargetsMap &StepStatus,
                             ContainerToTargetsMap &Results,
                             ArrayRef<string> Names) const {
  auto &OutputContainerTarget = Results[Names[PipeArgumentTargetIndex]];

  deduceResults(StepStatus, OutputContainerTarget, Names);
}

void Contract::deduceResults(ContainerToTargetsMap &StepStatus,
                             TargetsList &Results,
                             ArrayRef<string> Names) const {
  if (Source == nullptr) {
    PathComponents List;

    for (size_t I = 0; I < TargetKind->depth(); I++)
      List.emplace_back(PathComponent::all());

    Results.emplace_back(std::move(List), *TargetKind);
    return;
  }

  auto &SourceContainerTargets = StepStatus[Names[PipeArgumentSourceIndex]];

  const auto Matches = [this](const pipeline::Target &Input) {
    return forwardMatches(Input);
  };

  TargetsList Tmp;
  copy_if(SourceContainerTargets, back_inserter(Tmp), Matches);
  if (Preservation == pipeline::InputPreservation::Erase)
    erase_if(SourceContainerTargets, Matches);

  for (pipeline::Target &Target : Tmp)
    forward(Target);
  copy(Tmp, back_inserter(Results));
}

ContainerToTargetsMap
Contract::deduceRequirements(const ContainerToTargetsMap &Output,
                             ArrayRef<string> Names) const {

  ContainerToTargetsMap Requirements = Output;
  TargetsList &SourceContainer = Requirements[Names[PipeArgumentSourceIndex]];
  TargetsList &TargetContainer = Requirements[Names[PipeArgumentTargetIndex]];

  deduceRequirements(SourceContainer, TargetContainer);

  return Requirements;
}

void Contract::deduceRequirements(TargetsList &Source,
                                  TargetsList &Target) const {
  const auto Matches = [this](const pipeline::Target &Input) {
    if (this->Source == nullptr)
      return false;
    bool PreservedInput = Preservation == pipeline::InputPreservation::Preserve;
    return backwardMatches(Input) or (PreservedInput and forwardMatches(Input));
  };

  TargetsList Tmp;

  copy_if(Target, back_inserter(Tmp), Matches);

  // Transform the forward inputs/backward outputs that match,
  // they are trasformed by the current Pipe
  for (pipeline::Target &Out : Tmp)
    backward(Out);

  // Erase from the Target those that will produced by me
  erase_if(Target, Matches);

  copy(Tmp, back_inserter(Source));
}

void Contract::forward(pipeline::Target &Input) const {
  // A Pipe cannot yield a instance with multiple kinds when going
  // forward.
  revng_assert(Input.kindExactness() == Exactness::Exact);

  const auto *OutputKind = TargetKind != nullptr ? TargetKind :
                                                   &Input.getKind();
  Input.setKind(*OutputKind);
  forwardGranularity(Input);
}

bool Contract::forwardMatches(const Target &In) const {
  switch (InputContract) {
  case Exactness::DerivedFrom:
    return Source->ancestorOf(In.getKind());
  case Exactness::Exact:
    return &In.getKind() == Source;
  }
  return false;
}

void Contract::backward(Target &Output) const {
  if (not backwardMatches(Output))
    return;

  Output.setKind(backwardInputKind(Output));
  Output.setExactness(backwardInputContract(Output));
  backwardGranularity(Output);
}

Exactness::Values Contract::backwardInputContract(const Target &O) const {
  if (TargetKind != nullptr)
    return InputContract;

  if (InputContract == Exactness::Exact)
    return Exactness::Exact;

  return O.kindExactness();
}

void Contract::forwardGranularity(Target &Input) const {
  const auto *InputGranularity = &Source->granularity();
  const auto *OutputGranularity = TargetKind != nullptr ?
                                    &TargetKind->granularity() :
                                    InputGranularity;
  if (InputGranularity == OutputGranularity)
    return;

  // if the output is at a greater level of depth of the hierarchy
  // than the input, for each level of difference add a granularity to the
  // target.
  //
  if (InputGranularity->ancestorOf(*OutputGranularity)) {
    while (InputGranularity != OutputGranularity) {
      Input.addPathComponent();
      OutputGranularity = OutputGranularity->parent();
    }
    return;
  }

  // If the output is less fined grained than the input drop levels of
  // granularity until they have the same.
  if (OutputGranularity->ancestorOf(*InputGranularity)) {
    while (OutputGranularity != InputGranularity) {
      // if you are decreasing the granularity, you must have at your disposal
      // all symbols.
      revng_assert(Input.getPathComponents().back().isAll());
      Input.dropPathComponent();
      InputGranularity = InputGranularity->parent();
    }
    return;
  }

  revng_abort("Unreachable");
}

void Contract::backwardGranularity(Target &Out) const {
  const auto *InputGranularity = &Source->granularity();
  const auto *OutputGranularity = TargetKind != nullptr ?
                                    &TargetKind->granularity() :
                                    InputGranularity;
  if (InputGranularity == OutputGranularity)
    return;

  if (OutputGranularity->ancestorOf(*InputGranularity)) {
    while (InputGranularity != OutputGranularity) {
      Out.addPathComponent();
      InputGranularity = InputGranularity->parent();
    }
    return;
  }

  if (InputGranularity->ancestorOf(*OutputGranularity)) {
    while (InputGranularity != OutputGranularity) {
      // if you are decreasing the granularity, you must have at your disposal
      // all symbols.
      Out.dropPathComponent();
      OutputGranularity = OutputGranularity->parent();
    }
    return;
  }

  revng_abort("Unreachable");
}

const Kind &Contract::backwardInputKind(const Target &Output) const {
  // If the Pipe requires exactly a particular kind, return that one
  if (InputContract == Exactness::Exact)
    return *Source;

  if (TargetKind != nullptr)
    return *Source;

  // Otherwise return the most restricting between input requirement and
  // output. We have already know that one derives the other.
  if (Source->ancestorOf(Output.getKind()))
    return Output.getKind();

  return *Source;
}

bool Contract::backwardMatches(const Target &Out) const {
  if (TargetKind != nullptr)
    return &Out.getKind() == TargetKind;

  switch (InputContract) {
  case Exactness::DerivedFrom:
    return Source->ancestorOf(Out.getKind())
           or (Out.kindExactness() == Exactness::DerivedFrom
               and Out.getKind().ancestorOf(*Source));
  case Exactness::Exact:
    return Out.getKind().ancestorOf(*Source);
  }
}

using BCS = ContainerToTargetsMap;
bool Contract::forwardMatches(const BCS &StepStatus,
                              ArrayRef<string> Names) const {
  auto It = StepStatus.find(Names[PipeArgumentSourceIndex]);
  if (It == StepStatus.end())
    return false;
  const auto &SourceContainerTargets = It->second;

  const auto Matches = [this](const pipeline::Target &Input) {
    return forwardMatches(Input);
  };

  return any_of(SourceContainerTargets, Matches);
}

bool Contract::backwardMatches(const BCS &StepStatus,
                               ArrayRef<string> Names) const {
  auto It = StepStatus.find(Names[PipeArgumentTargetIndex]);
  if (It == StepStatus.end())
    return false;
  const auto &OutputContainerTarget = It->second;

  const auto Matches = [this](const pipeline::Target &Input) {
    bool PreservedInput = Preservation == pipeline::InputPreservation::Preserve;
    return backwardMatches(Input) or (PreservedInput and forwardMatches(Input));
  };

  return any_of(OutputContainerTarget, Matches);
}

void Contract::insertDefaultInput(BCS &Status, ArrayRef<string> Names) const {
  auto &SourceContainerTargets = Status[Names[PipeArgumentSourceIndex]];

  llvm::SmallVector<PathComponent, 4> PathComponents(Source->depth() + 1,
                                                     PathComponent::all());

  Target Target(move(PathComponents), *Source, InputContract);
  SourceContainerTargets.push_back(move(Target));
}

bool ContractGroup::forwardMatches(const BCS &Status,
                                   llvm::ArrayRef<std::string> Names) const {
  return all_of(Content, [&Status, &Names](const auto &C) {
    return C.forwardMatches(Status, Names);
  });
}

bool ContractGroup::backwardMatches(const BCS &Status,
                                    llvm::ArrayRef<std::string> Names) const {
  return any_of(Content, [&Status, &Names](const auto &C) {
    return C.backwardMatches(Status, Names);
  });
}

ContainerToTargetsMap
ContractGroup::deduceRequirements(const ContainerToTargetsMap &StepStatus,
                                  ArrayRef<string> Names) const {
  if (not backwardMatches(StepStatus, Names)) {
    return StepStatus;
  }

  ContainerToTargetsMap Results;
  for (const auto &C : llvm::reverse(Content)) {
    if (C.backwardMatches(StepStatus, Names))
      Results.merge(C.deduceRequirements(StepStatus, Names));
    else
      C.insertDefaultInput(Results, Names);
  }

  return Results;
}

void ContractGroup::deduceResults(ContainerToTargetsMap &StepStatus,
                                  ArrayRef<string> Names) const {
  if (not forwardMatches(StepStatus, Names))
    return;

  ContainerToTargetsMap Results;
  for (const auto &C : Content)
    C.deduceResults(StepStatus, Results, Names);

  StepStatus.merge(Results);
}
