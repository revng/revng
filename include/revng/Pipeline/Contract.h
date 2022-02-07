#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Pipeline/Target.h"

namespace pipeline {

namespace InputPreservation {
enum Values { Erase, Preserve };
}

/// A contract enstablishes a what operations a pipe can perform on input
/// container and output container both referred by their index.
///
/// Before explaining more formally how contracts operates it's better consider
/// a couple of examples
///
/// A contract from root to binary (both kinds with granulartiy 1), requires
/// that the pipe it is attached to will convert all roots with pathComponents
/// (i) into binaries with pathComponents (i)
///
/// A contract from root(granularity 1) to function(granularity 2), requires
/// that the pipe it is attached to will convert all roots with pathComponents
/// (i) into functions with pathComponents(i, *).
///
/// A contract from functions(granularity 2) to root(granularity 1), requires
/// that the pipe it is attached to will convert all functions with
/// pathComponents (i, *) into roots with pathComponents(i). Furthermore it is
/// guaranteed that only functions in the form (i, *) are presents, in other
/// words that either all or none functions are forwarded to the pipe.
///
/// A pipe marked with a single contract must transform all and only items from
/// the input container that have Targets with the same kind as if SourceKind
/// (or a derived kind if Exactness is not Exact)
///
/// The pipe must furthermore esures that for each item in the input container
/// that matches the requirements another target is created in the target
/// container (is the target container exists), such that
///
/// If the output kind has a greater granularity than the inputkind then for all
/// Targets in input with source kind and pathComponents (i1, ..., in) then a
/// target with kind output and pathComponents (i1, ..., in, *) exists.
///
/// If the output has a smaller granularity g1 than the input granularity g2
/// then for the pipe must operate on all targets with Kind source and
/// pathcomponents(i1, ..., i_g2) and yield a a target for each with Kind target
/// and pathComponents(i1, ..., i_g1), it will be ensured that when the pipe
/// executed each target in the input container that must be transformed will be
/// in the form (i1, ..., i_g2) where each pathcomponents i_x such that g1 <= x
/// <= g2 is *.
///
/// A contract must refer to a source target containers than can be the same.
/// The index refers to the list of names passed in the various methods of this
/// class.
///
/// In practice they refer to the argument index of the run method of pipes,
/// ignoring argument zero which is always Context.
class Contract {
public:
  static constexpr auto Erase = InputPreservation::Erase;

private:
  const Kind *Source;
  Exactness::Values InputContract;
  const Kind *TargetKind;
  size_t PipeArgumentSourceIndex;
  size_t PipeArgumentTargetIndex;
  InputPreservation::Values Preservation;

public:
  constexpr Contract(const Kind &Source,
                     Exactness::Values InputContract,
                     size_t PipeArgumentSourceIndex,
                     const Kind &Target,
                     size_t PipeArgumentTargetIndex = 0,
                     InputPreservation::Values Preserve = Erase) :
    Source(&Source),
    InputContract(InputContract),
    TargetKind(&Target),
    PipeArgumentSourceIndex(PipeArgumentSourceIndex),
    PipeArgumentTargetIndex(PipeArgumentTargetIndex),
    Preservation(Preserve) {}

  constexpr Contract(const Kind &Target, size_t PipeArgumentTargetIndex = 0) :
    Source(nullptr),
    InputContract(Exactness::Exact),
    TargetKind(&Target),
    PipeArgumentSourceIndex(0),
    PipeArgumentTargetIndex(PipeArgumentTargetIndex),
    Preservation(Erase) {}

  constexpr Contract(const Kind &Source,
                     Exactness::Values InputContract,
                     size_t PipeArgumentSourceIndex = 0,
                     InputPreservation::Values Preserve = Erase) :
    Source(&Source),
    InputContract(InputContract),
    TargetKind(nullptr),
    PipeArgumentSourceIndex(PipeArgumentSourceIndex),
    PipeArgumentTargetIndex(PipeArgumentSourceIndex),
    Preservation(Preserve) {}

public:
  void deduceResults(ContainerToTargetsMap &StepStatus,
                     llvm::ArrayRef<std::string> ContainerNames) const;

  void deduceResults(ContainerToTargetsMap &StepStatus,
                     TargetsList &Results,
                     llvm::ArrayRef<std::string> ContainerNames) const;

  void deduceResults(ContainerToTargetsMap &StepStatus,
                     ContainerToTargetsMap &Results,
                     llvm::ArrayRef<std::string> ContainerNames) const;

  ContainerToTargetsMap
  deduceRequirements(const ContainerToTargetsMap &PipeOutput,
                     llvm::ArrayRef<std::string> ContainerNames) const;

  bool forwardMatches(const ContainerToTargetsMap &Status,
                      llvm::ArrayRef<std::string> ContainerNames) const;
  bool backwardMatches(const ContainerToTargetsMap &Status,
                       llvm::ArrayRef<std::string> ContainerNames) const;

  void insertDefaultInput(ContainerToTargetsMap &Status,
                          llvm::ArrayRef<std::string> ContainerNames) const;

private:
  void forward(Target &Input) const;
  bool forwardMatches(const Target &Input) const;
  void forwardRank(Target &Input) const;

  /// Target fixed -> Output must be exactly Target.
  /// Target same as Source, Source derived from base -> Most strict between
  /// source and target Target same as source, source exactly base -> base.
  void backward(Target &Output) const;
  Exactness::Values backwardInputContract(const Target &Output) const;
  void backwardRank(Target &Output) const;
  const Kind &backwardInputKind(const Target &Output) const;
  bool backwardMatches(const Target &Output) const;

  /// Target is the container in which the Pipe would write when used to produce
  /// the targets.
  void deduceRequirements(TargetsList &SourceContainer,
                          TargetsList &TargetContainer) const;
};

/// A group contract is the way contracts can be composed.
/// All subcontracts inside a group contract are evaluated in parallel on the
/// same input and their, and their output is merged.
class ContractGroup {
public:
  static constexpr auto Erase = InputPreservation::Erase;

private:
  llvm::SmallVector<Contract, 2> Content;

public:
  ContractGroup(std::initializer_list<Contract> Content) :
    Content(std::move(Content)) {}

  ContractGroup(llvm::SmallVector<Contract, 2> Content) :
    Content(std::move(Content)) {}

  ContractGroup(llvm::ArrayRef<ContractGroup> Contracts) {
    for (const auto &C : Contracts)
      for (const auto &Entry : C.Content)
        Content.push_back(Entry);
  }

  ContractGroup(const Kind &Source,
                Exactness::Values InputContract,
                size_t PipeArgumentSourceIndex,
                const Kind &Target,
                size_t PipeArgumentTargetIndex = 0,
                InputPreservation::Values Preservation = Erase) :
    Content({ Contract(Source,
                       InputContract,
                       PipeArgumentSourceIndex,
                       Target,
                       PipeArgumentTargetIndex,
                       Preservation) }) {}

  ContractGroup(const Kind &Source,
                Exactness::Values InputContract,
                size_t PipeArgumentSourceIndex = 0,
                InputPreservation::Values Preservation = Erase) :
    Content({ Contract(Source,
                       InputContract,
                       PipeArgumentSourceIndex,
                       Preservation) }) {}

  ContractGroup(const Kind &Target, size_t PipeArgumentTargetIndex = 0) :
    Content({ Contract(Target, PipeArgumentTargetIndex) }) {}

  static ContractGroup
  transformOnlyArgument(const Kind &Source,
                        Exactness::Values Exact,
                        const Kind &Target,
                        InputPreservation::Values Preservation) {
    return ContractGroup(Source, Exact, 0, Target, 0, Preservation);
  }

public:
  [[nodiscard]] ContainerToTargetsMap
  deduceRequirements(const ContainerToTargetsMap &StepStatus,
                     llvm::ArrayRef<std::string> ContainerNames) const;
  void deduceResults(ContainerToTargetsMap &StepStatus,
                     llvm::ArrayRef<std::string> ContainerNames) const;

  bool forwardMatches(const ContainerToTargetsMap &Status,
                      llvm::ArrayRef<std::string> ContainerNames) const;
  bool backwardMatches(const ContainerToTargetsMap &Status,
                       llvm::ArrayRef<std::string> ContainerNames) const;
};

} // namespace pipeline
