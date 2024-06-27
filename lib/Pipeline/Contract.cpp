/// \file Contract.cpp
/// A contract is "Rule" attached to a pipe that specifies what kind of
/// transformations the pipe is allowed to do on input containers.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Target.h"

using namespace pipeline;
using namespace llvm;
using namespace std;

void Contract::deduceResults(const Context &Ctx,
                             ContainerToTargetsMap &StepStatus,
                             ArrayRef<string> Names) const {
  auto &OutputContainerTarget = StepStatus[Names[PipeArgumentTargetIndex]];

  TargetsList Tmp;
  deduceResults(Ctx, StepStatus, Tmp, Names);

  copy(Tmp, back_inserter(OutputContainerTarget));
}

void Contract::deduceResults(const Context &Ctx,
                             ContainerToTargetsMap &StepStatus,
                             ContainerToTargetsMap &Results,
                             ArrayRef<string> Names) const {
  auto &OutputContainerTarget = Results[Names[PipeArgumentTargetIndex]];

  deduceResults(Ctx, StepStatus, OutputContainerTarget, Names);
}

static TargetsList copyEntriesOfKind(const TargetsList &List, const Kind &K) {
  auto Range = List.filterByKind(K);
  TargetsList ToReturn;
  copy(Range.begin(), Range.end(), std::back_inserter(ToReturn));
  return ToReturn;
}

static TargetsList extracEntriesOfKind(TargetsList &List, const Kind &K) {
  auto Range = List.filterByKind(K);
  TargetsList ToReturn;
  copy(Range.begin(), Range.end(), std::back_inserter(ToReturn));
  List.erase(Range.begin(), Range.end());
  return ToReturn;
}

void Contract::deduceResults(const Context &Ctx,
                             ContainerToTargetsMap &StepStatus,
                             TargetsList &Results,
                             ArrayRef<string> Names) const {
  if (Source == nullptr) {
    TargetKind->appendAllTargets(Ctx, Results);
    return;
  }

  auto &SourceContainerTargets = StepStatus[Names[PipeArgumentSourceIndex]];
  auto Kinds = SourceContainerTargets.getContainedKinds();

  for (auto &Kind : Kinds) {

    auto Targets = Preservation == pipeline::InputPreservation::Erase ?
                     extracEntriesOfKind(SourceContainerTargets, *Kind) :
                     copyEntriesOfKind(SourceContainerTargets, *Kind);
    Targets = forward(Ctx, std::move(Targets));
    copy(Targets, back_inserter(Results));
  }
}

TargetsList Contract::forward(const Context &Ctx, TargetsList Input) const {
  if (Input.empty())
    return {};

  auto *Kind = &Input.front().getKind();
  if (Kind != Source)
    return Input;

  if (Source->depth() == TargetKind->depth()) {
    for (auto &Target : Input)
      Target.setKind(*TargetKind);
    return Input;
  }

  if (Source->depth() > TargetKind->depth()) {
    TargetsList All;
    TargetKind->appendAllTargets(Ctx, All);
    return All;
  }

  TargetsList All;
  Source->appendAllTargets(Ctx, All);
  if (All != Input) {
    return Input;
  }

  TargetsList AllDestination;
  TargetKind->appendAllTargets(Ctx, AllDestination);
  return AllDestination;
}

ContainerToTargetsMap
Contract::deduceRequirements(const Context &Ctx,
                             const ContainerToTargetsMap &Output,
                             ArrayRef<string> Names) const {

  ContainerToTargetsMap Requirements = Output;
  TargetsList &SourceContainer = Requirements[Names[PipeArgumentSourceIndex]];
  TargetsList &TargetContainer = Requirements[Names[PipeArgumentTargetIndex]];

  deduceRequirements(Ctx, SourceContainer, TargetContainer);

  return Requirements;
}

void Contract::deduceRequirements(const Context &Ctx,
                                  TargetsList &Source,
                                  TargetsList &Target) const {
  if (this->Source == nullptr) {
    return;
  }

  auto Kinds = Target.getContainedKinds();

  for (auto &Kind : Kinds) {
    if (Kind != TargetKind)
      continue;

    bool PreservedInput = Preservation == pipeline::InputPreservation::Preserve;
    auto Targets = extracEntriesOfKind(Target, *Kind);

    // Transform the forward inputs/backward outputs that match,
    // they are transformed by the current Pipe
    Targets = backward(Ctx, std::move(Targets));

    copy(Targets, back_inserter(Source));
  }
}

bool Contract::forwardMatches(const Context &Ctx, const TargetsList &In) const {
  if (Source == nullptr)
    return true;

  if (In.empty())
    return false;

  if (Source->depth() > TargetKind->depth()) {
    TargetsList All;
    return In.contains(All);
  }

  auto Kinds = In.getContainedKinds();
  auto Res = llvm::find(Kinds, Source) != Kinds.end();
  return Res;
}

TargetsList Contract::backward(const Context &Ctx, TargetsList Output) const {
  if (Output.empty())
    return {};

  auto Kind = &Output.front().getKind();

  if (Source->depth() == TargetKind->depth()) {
    for (auto &Target : Output)
      Target.setKind(*Source);
    return Output;
  }

  if (Source->depth() < TargetKind->depth()) {
    TargetsList All;
    Source->appendAllTargets(Ctx, All);
    return All;
  }

  if (Source->depth() > TargetKind->depth()) {
    TargetsList All;
    TargetKind->appendAllTargets(Ctx, All);
    if (All != Output) {
      return Output;
    }

    TargetsList AllDestination;
    Source->appendAllTargets(Ctx, AllDestination);
    return AllDestination;
  }

  return Output;
}

using BCS = ContainerToTargetsMap;
bool Contract::forwardMatches(const Context &Ctx,
                              const BCS &StepStatus,
                              ArrayRef<string> Names) const {
  auto It = StepStatus.find(Names[PipeArgumentSourceIndex]);
  if (It == StepStatus.end())
    return false;
  const auto &SourceContainerTargets = It->second;
  return forwardMatches(Ctx, SourceContainerTargets);
}

bool Contract::backwardMatchesImpl(const Context &Ctx,
                                   const TargetsList &List) const {
  if (Source == nullptr)
    return true;

  if (List.empty())
    return false;

  if (Source->depth() == TargetKind->depth()) {
    auto Kinds = List.getContainedKinds();
    return find(Kinds, this->TargetKind) != Kinds.end();
  }

  return true;
}

bool Contract::backwardMatches(const Context &Ctx,
                               const BCS &StepStatus,
                               ArrayRef<string> Names) const {
  auto It = StepStatus.find(Names[PipeArgumentTargetIndex]);
  if (It == StepStatus.end())
    return false;
  const auto &OutputContainerTarget = It->second;

  bool PreservedInput = Preservation == pipeline::InputPreservation::Preserve;
  return backwardMatchesImpl(Ctx, OutputContainerTarget)
         or (PreservedInput and forwardMatches(Ctx, OutputContainerTarget));
}

void Contract::insertDefaultInput(const Context &Ctx,
                                  BCS &Status,
                                  ArrayRef<string> Names) const {
  if (Source == nullptr)
    return;
  auto &SourceContainerTargets = Status[Names[PipeArgumentSourceIndex]];
  Source->appendAllTargets(Ctx, SourceContainerTargets);
}

bool ContractGroup::forwardMatches(const Context &Ctx,
                                   const BCS &Status,
                                   llvm::ArrayRef<std::string> Names) const {
  return all_of(Content, [&](const auto &C) {
    return C.forwardMatches(Ctx, Status, Names);
  });
}

bool ContractGroup::backwardMatches(const Context &Ctx,
                                    const BCS &Status,
                                    llvm::ArrayRef<std::string> Names) const {
  return any_of(Content, [&](const auto &C) {
    return C.backwardMatches(Ctx, Status, Names);
  });
}

ContainerToTargetsMap
ContractGroup::deduceRequirements(const Context &Ctx,
                                  const ContainerToTargetsMap &StepStatus,
                                  ArrayRef<string> Names) const {
  if (not backwardMatches(Ctx, StepStatus, Names)) {
    return StepStatus;
  }

  ContainerToTargetsMap Results;
  for (const auto &C : llvm::reverse(Content)) {
    if (C.backwardMatches(Ctx, StepStatus, Names)) {

      Results.merge(C.deduceRequirements(Ctx, StepStatus, Names));
    } else {

      C.insertDefaultInput(Ctx, Results, Names);
    }
  }

  return Results;
}

void ContractGroup::deduceResults(const Context &Ctx,
                                  ContainerToTargetsMap &StepStatus,
                                  ArrayRef<string> Names) const {
  if (not forwardMatches(Ctx, StepStatus, Names))
    return;

  ContainerToTargetsMap Results;
  for (const auto &C : Content)
    C.deduceResults(Ctx, StepStatus, Results, Names);

  StepStatus.merge(Results);
}
