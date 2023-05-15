#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Invokable.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Debug.h"

namespace pipeline {

namespace detail {

template<typename T>
concept HasContract = requires(T P) {
  { llvm::ArrayRef<ContractGroup>(P.getContract()) };
};

template<typename T, typename FirstRunArg, typename... Args>
concept Pipe = Invokable<T, FirstRunArg, Args...>
               and (IsContainer<Args> and ...) and HasContract<T>;

template<typename T>
concept HasPrecondition = requires(const T &P) {
  { P.checkPrecondition };
};

template<typename C, typename First, typename... Rest>
constexpr bool checkPipe(auto (C::*)(First, Rest...))
  requires Pipe<C, First, Rest...>
{
  return true;
}

template<typename PipeType>
class PipeWrapperImpl;

class PipeWrapperBase : public InvokableWrapperBase {
public:
  template<typename PipeType>
  using ImplType = PipeWrapperImpl<PipeType>;

public:
  virtual ContainerToTargetsMap
  getRequirements(const Context &Ctx,
                  const ContainerToTargetsMap &Target) const = 0;
  virtual ContainerToTargetsMap
  deduceResults(const Context &Ctx, ContainerToTargetsMap &Target) const = 0;
  virtual bool areRequirementsMet(const Context &Ctx,
                                  const ContainerToTargetsMap &Input) const = 0;
  virtual std::unique_ptr<PipeWrapperBase>
  clone(std::vector<std::string> NewRunningContainersNames = {}) const = 0;
  virtual llvm::Error checkPrecondition(const Context &Ctx) const = 0;

  virtual ~PipeWrapperBase() = default;
};

/// A pipe must be type erased somehow to become compatible with a pipeline,
/// a PipeWrapperImpl takes care of this issue, it can be constructed from
/// any pipeline type, and it will expose the contract and run method of that
/// enforcer.
template<typename PipeType>
class PipeWrapperImpl : public PipeWrapperBase {
private:
  static constexpr bool CheckPipe = checkPipe(&PipeType::run);
  static_assert(CheckPipe);

  InvokableWrapperImpl<PipeType> Invokable;

public:
  PipeWrapperImpl(PipeType ActualPipe,
                  std::vector<std::string> RunningContainersNames) :
    Invokable(std::move(ActualPipe), std::move(RunningContainersNames)) {}

  PipeWrapperImpl(const PipeWrapperImpl &ActualPipe,
                  std::vector<std::string> RunningContainersNames) :
    Invokable(ActualPipe.Invokable, std::move(RunningContainersNames)) {}

  PipeWrapperImpl(PipeWrapperImpl &&ActualPipe,
                  std::vector<std::string> RunningContainersNames) :
    Invokable(std::move(ActualPipe.Invokable),
              std::move(RunningContainersNames)) {}

public:
  bool areRequirementsMet(const Context &Ctx,
                          const ContainerToTargetsMap &Input) const override {
    const auto &Contracts = Invokable.getPipe().getContract();
    if (Contracts.size() == 0)
      return true;

    ContainerToTargetsMap ToCheck = Input;
    for (const auto &Contract : Contracts) {
      if (Contract.forwardMatches(Ctx,
                                  ToCheck,
                                  Invokable.getRunningContainersNames()))
        return true;

      Contract.deduceResults(Ctx,
                             ToCheck,
                             Invokable.getRunningContainersNames());
    }

    return false;
  }

  ContainerToTargetsMap
  getRequirements(const Context &Ctx,
                  const ContainerToTargetsMap &Target) const override {
    const auto &Contracts = Invokable.getPipe().getContract();
    auto ToReturn = Target;
    for (const auto &Contract : llvm::reverse(Contracts))
      ToReturn = Contract
                   .deduceRequirements(Ctx,
                                       ToReturn,
                                       Invokable.getRunningContainersNames());
    return ToReturn;
  }

  ContainerToTargetsMap
  deduceResults(const Context &Ctx,
                ContainerToTargetsMap &Target) const override {
    const auto &Contracts = Invokable.getPipe().getContract();
    for (const auto &Contract : Contracts)
      Contract.deduceResults(Ctx,
                             Target,
                             Invokable.getRunningContainersNames());
    return Target;
  }

  std::unique_ptr<PipeWrapperBase>
  clone(std::vector<std::string> NewContainersNames = {}) const override {
    if (NewContainersNames.empty())
      return std::make_unique<PipeWrapperImpl>(*this);
    return std::make_unique<PipeWrapperImpl>(*this,
                                             std::move(NewContainersNames));
  }

  llvm::Error checkPrecondition(const Context &Ctx) const override {
    if constexpr (not HasPrecondition<PipeType>)
      return llvm::Error::success();
    else
      return Invokable.getPipe().checkPrecondition(Ctx);
  }

public:
  void dump(std::ostream &OS,
            size_t Indentation) const override debug_function {
    Invokable.dump(OS, Indentation);
  }

  void print(const Context &Ctx,
             llvm::raw_ostream &OS,
             size_t Indentation) const override {
    Invokable.print(Ctx, OS, Indentation);
  }

  llvm::Error run(Context &Ctx,
                  ContainerSet &Containers,
                  const llvm::StringMap<std::string> &ExtraArgs) override {
    return Invokable.run(Ctx, Containers, ExtraArgs);
  }

  std::vector<std::string> getOptionsNames() const override {
    return Invokable.getOptionsNames();
  }

  std::vector<std::string> getOptionsTypes() const override {
    return Invokable.getOptionsTypes();
  }

  std::vector<std::string> getRunningContainersNames() const override {
    return Invokable.getRunningContainersNames();
  }
  bool isContainerArgumentConst(size_t ArgumentIndex) const override {
    return Invokable.isContainerArgumentConst(ArgumentIndex);
  }
  std::string getName() const override { return Invokable.getName(); }
};

} // namespace detail

using PipeWrapper = InvokableWrapper<detail::PipeWrapperBase>;

} // namespace pipeline
