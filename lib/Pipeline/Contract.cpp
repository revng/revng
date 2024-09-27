/// \file Contract.cpp
/// A contract is "Rule" attached to a pipe that specifies what kind of
/// transformations the pipe is allowed to do on input containers.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Target.h"

using namespace pipeline;
using namespace llvm;
using namespace std;

void Contract::deduceResults(const Context &Context,
                             ContainerToTargetsMap &StepStatus,
                             ArrayRef<string> Names) const {
  auto &OutputContainerTarget = StepStatus[Names[PipeArgumentTargetIndex]];

  TargetsList Tmp;
  deduceResults(Context, StepStatus, Tmp, Names);

  copy(Tmp, back_inserter(OutputContainerTarget));
}

void Contract::deduceResults(const Context &Context,
                             ContainerToTargetsMap &StepStatus,
                             ContainerToTargetsMap &Results,
                             ArrayRef<string> Names) const {
  auto &OutputContainerTarget = Results[Names[PipeArgumentTargetIndex]];

  deduceResults(Context, StepStatus, OutputContainerTarget, Names);
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

void Contract::deduceResults(const Context &Context,
                             ContainerToTargetsMap &StepStatus,
                             TargetsList &Results,
                             ArrayRef<string> Names) const {
  if (Source == nullptr) {
    TargetKind->appendAllTargets(Context, Results);
    return;
  }

  auto &SourceContainerTargets = StepStatus[Names[PipeArgumentSourceIndex]];
  auto Kinds = SourceContainerTargets.getContainedKinds();

  for (auto &Kind : Kinds) {

    auto Targets = Preservation == pipeline::InputPreservation::Erase ?
                     extracEntriesOfKind(SourceContainerTargets, *Kind) :
                     copyEntriesOfKind(SourceContainerTargets, *Kind);
    Targets = forward(Context, std::move(Targets));
    copy(Targets, back_inserter(Results));
  }
}

TargetsList Contract::forward(const Context &Context, TargetsList Input) const {
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
    TargetKind->appendAllTargets(Context, All);
    return All;
  }

  TargetsList All;
  Source->appendAllTargets(Context, All);
  if (All != Input) {
    return Input;
  }

  TargetsList AllDestination;
  TargetKind->appendAllTargets(Context, AllDestination);
  return AllDestination;
}

ContainerToTargetsMap
Contract::deduceRequirements(const Context &Context,
                             const ContainerToTargetsMap &Output,
                             ArrayRef<string> Names) const {

  ContainerToTargetsMap Requirements = Output;
  TargetsList &SourceContainer = Requirements[Names[PipeArgumentSourceIndex]];
  TargetsList &TargetContainer = Requirements[Names[PipeArgumentTargetIndex]];

  deduceRequirements(Context, SourceContainer, TargetContainer);

  return Requirements;
}

void Contract::deduceRequirements(const Context &Context,
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
    Targets = backward(Context, std::move(Targets));

    copy(Targets, back_inserter(Source));
  }
}

bool Contract::forwardMatches(const TargetsList &In) const {
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

TargetsList Contract::backward(const Context &Context,
                               TargetsList Output) const {
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
    Source->appendAllTargets(Context, All);
    return All;
  }

  if (Source->depth() > TargetKind->depth()) {
    TargetsList All;
    TargetKind->appendAllTargets(Context, All);
    if (All != Output) {
      return Output;
    }

    TargetsList AllDestination;
    Source->appendAllTargets(Context, AllDestination);
    return AllDestination;
  }

  return Output;
}

using BCS = ContainerToTargetsMap;
bool Contract::forwardMatches(const BCS &StepStatus,
                              ArrayRef<string> Names) const {
  auto It = StepStatus.find(Names[PipeArgumentSourceIndex]);
  if (It == StepStatus.end())
    return false;
  const auto &SourceContainerTargets = It->second;
  return forwardMatches(SourceContainerTargets);
}

bool Contract::backwardMatchesImpl(const TargetsList &List) const {
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

bool Contract::backwardMatches(const BCS &StepStatus,
                               ArrayRef<string> Names) const {
  auto It = StepStatus.find(Names[PipeArgumentTargetIndex]);
  if (It == StepStatus.end())
    return false;
  const auto &OutputContainerTarget = It->second;

  bool PreservedInput = Preservation == pipeline::InputPreservation::Preserve;
  return backwardMatchesImpl(OutputContainerTarget)
         or (PreservedInput and forwardMatches(OutputContainerTarget));
}

void Contract::insertDefaultInput(const Context &Context,
                                  BCS &Status,
                                  ArrayRef<string> Names) const {
  if (Source == nullptr)
    return;
  auto &SourceContainerTargets = Status[Names[PipeArgumentSourceIndex]];
  Source->appendAllTargets(Context, SourceContainerTargets);
}

bool ContractGroup::forwardMatches(const BCS &Status,
                                   llvm::ArrayRef<std::string> Names) const {
  return all_of(Content,
                [&](const auto &C) { return C.forwardMatches(Status, Names); });
}

bool ContractGroup::backwardMatches(const BCS &Status,
                                    llvm::ArrayRef<std::string> Names) const {
  return any_of(Content, [&](const auto &C) {
    return C.backwardMatches(Status, Names);
  });
}

std::pair<size_t, const Kind *> Contract::getOutput() const {
  return { this->PipeArgumentTargetIndex, this->TargetKind };
}

llvm::SmallVector<std::pair<size_t, const Kind *>, 4>
ContractGroup::getOutputs() const {
  llvm::SmallVector<std::pair<size_t, const Kind *>, 4> Result;
  for (const Contract &C : Content) {
    Result.push_back(C.getOutput());
  }
  return Result;
}

ContainerToTargetsMap
ContractGroup::deduceRequirements(const Context &Context,
                                  const ContainerToTargetsMap &StepStatus,
                                  ArrayRef<string> Names) const {
  if (not backwardMatches(StepStatus, Names)) {
    return StepStatus;
  }

  ContainerToTargetsMap Results;
  for (const auto &C : llvm::reverse(Content)) {
    if (C.backwardMatches(StepStatus, Names)) {

      Results.merge(C.deduceRequirements(Context, StepStatus, Names));
    } else {

      C.insertDefaultInput(Context, Results, Names);
    }
  }

  return Results;
}

void ContractGroup::deduceResults(const Context &Context,
                                  ContainerToTargetsMap &StepStatus,
                                  ArrayRef<string> Names) const {
  if (not forwardMatches(StepStatus, Names))
    return;

  ContainerToTargetsMap Results;
  for (const auto &C : Content)
    C.deduceResults(Context, StepStatus, Results, Names);

  StepStatus.merge(Results);
}
