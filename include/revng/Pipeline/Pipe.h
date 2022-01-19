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
#include "revng/Support/Debug.h"

namespace pipeline {

template<typename T>
concept IsContext = std::is_convertible_v<T, Context &>;

template<typename T>
concept IsConstContext = std::is_convertible_v<T, const Context &>;

template<typename T>
concept IsContainer = std::is_base_of_v<ContainerBase, std::decay_t<T>>;

template<typename T>
concept HasName = requires() {
  { T::Name } -> convertible_to<const char *>;
};

template<typename T>
concept HasContract = requires(T a) {
  { llvm::ArrayRef<ContractGroup>(a.getContract()) };
};

/// A Pipe is a class with the following characteristics:
///
/// * It must have a static constexpr field named Name that is a string
///   describing its name. Mostly used for debug purposes.
/// * A std::array<ContractGroup, X> getContract() const method that returns
///   the contract of such class.
/// * a void run(T...) method where the first argument must be a Context and
///   every other type must be the most derived type of a Container.
///   The operation performed by run MUST be consisted with the contract.
template<typename PipeType, typename FirstRunArg, typename... Rest>
concept Pipe =
  HasName<PipeType> and HasContract<PipeType> and(IsContainer<Rest> and...)
  and (IsContext<FirstRunArg> || IsConstContext<FirstRunArg>);

namespace detail {
using StringArrayRef = llvm::ArrayRef<std::string>;

template<typename T>
auto &getContainerFromName(ContainerSet &Containers, llvm::StringRef Name) {
  return llvm::cast<T>(Containers[Name]);
}

template<typename PipeType, typename... Args, size_t... S>
void invokeImpl(Context &Ctx,
                PipeType &Pipe,
                void (PipeType::*F)(Context &, Args...),
                ContainerSet &Containers,
                const StringArrayRef &ArgsNames,
                const std::integer_sequence<size_t, S...> &) {

  using namespace std;
  (Pipe.*F)(Ctx,
            getContainerFromName<decay_t<Args>>(Containers, ArgsNames[S])...);
}

template<typename PipeType, typename... Args, size_t... S>
void invokeImpl(const Context &Ctx,
                PipeType &Pipe,
                void (PipeType::*F)(const Context &, Args...),
                ContainerSet &Containers,
                const StringArrayRef &ArgsNames,
                const std::integer_sequence<size_t, S...> &) {
  using namespace std;
  (Pipe.*F)(Ctx,
            getContainerFromName<decay_t<Args>>(Containers, ArgsNames[S])...);
}

/// Invokes the F member function on the Pipe Pipe passing as nth argument the
/// container with the name equal to the nth element of ArgsNames.
template<typename PipeType, typename... Args>
void invokePipeFunction(Context &Ctx,
                        PipeType &Pipe,
                        void (PipeType::*F)(Context &, Args...),
                        ContainerSet &Containers,
                        const llvm::ArrayRef<std::string> &ArgsNames) {
  constexpr auto
    Indexes = std::make_integer_sequence<size_t, sizeof...(Args)>();
  revng_assert(sizeof...(Args) == ArgsNames.size());
  invokeImpl(Ctx, Pipe, F, Containers, ArgsNames, Indexes);
}

template<typename PipeType, typename... Args>
void invokePipeFunction(Context &Ctx,
                        PipeType &Pipe,
                        void (PipeType::*F)(const Context &, Args...),
                        ContainerSet &Containers,
                        const llvm::ArrayRef<std::string> &ArgsNames) {
  constexpr auto
    Indexes = std::make_integer_sequence<size_t, sizeof...(Args)>();
  revng_assert(sizeof...(Args) == ArgsNames.size());
  invokeImpl(Ctx, Pipe, F, Containers, ArgsNames, Indexes);
}

template<typename C, typename First, typename... Rest>
constexpr bool
checkPipe(void (C::*)(First, Rest...)) requires Pipe<C, First, Rest...> {
  return true;
}

class PipeWrapperBase {
public:
  virtual void run(Context &Ctx, ContainerSet &Containers) = 0;
  virtual ContainerToTargetsMap
  getRequirements(const ContainerToTargetsMap &Target) const = 0;
  virtual ContainerToTargetsMap
  deduceResults(ContainerToTargetsMap &Target) const = 0;
  virtual ~PipeWrapperBase() = default;
  virtual std::unique_ptr<PipeWrapperBase> clone() const = 0;
  virtual std::vector<std::string> getRunningContainersNames() const = 0;
  virtual std::string getName() const = 0;
  virtual void dump(std::ostream &OS, size_t indents) const = 0;
  virtual bool areRequirementsMet(const ContainerToTargetsMap &Input) const = 0;
};

template<typename T>
concept Dumpable = requires(T a) {
  { a.dump(dbg, 0) };
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

private:
  PipeType ActualPipe;
  std::vector<std::string> RunningContainersNames;

public:
  PipeWrapperImpl(PipeType ActualPipe,
                  std::vector<std::string> RunningContainersNames) :
    ActualPipe(std::move(ActualPipe)),
    RunningContainersNames(std::move(RunningContainersNames)) {}

  ~PipeWrapperImpl() override = default;

public:
  std::string getName() const override { return PipeType::Name; }

public:
  void run(Context &Ctx, ContainerSet &Containers) override {
    invokePipeFunction(Ctx,
                       ActualPipe,
                       &PipeType::run,
                       Containers,
                       RunningContainersNames);
  }

public:
  bool areRequirementsMet(const ContainerToTargetsMap &Input) const override {
    const auto &Contracts = ActualPipe.getContract();
    if (Contracts.size() == 0)
      return true;

    ContainerToTargetsMap ToCheck = Input;
    for (const auto &Contract : Contracts) {
      if (Contract.forwardMatches(ToCheck, RunningContainersNames))
        return true;

      Contract.deduceResults(ToCheck, RunningContainersNames);
    }

    return false;
  }

  ContainerToTargetsMap
  getRequirements(const ContainerToTargetsMap &Target) const override {
    const auto &Contracts = ActualPipe.getContract();
    auto ToReturn = Target;
    for (const auto &Contract : llvm::reverse(Contracts))
      ToReturn = Contract.deduceRequirements(ToReturn, RunningContainersNames);
    return ToReturn;
  }

  ContainerToTargetsMap
  deduceResults(ContainerToTargetsMap &Target) const override {
    const auto &Contracts = ActualPipe.getContract();
    for (const auto &Contract : Contracts)
      Contract.deduceResults(Target, RunningContainersNames);
    return Target;
  }

  std::unique_ptr<PipeWrapperBase> clone() const override {
    return std::make_unique<PipeWrapperImpl>(*this);
  }

  std::vector<std::string> getRunningContainersNames() const override {
    return RunningContainersNames;
  }

public:
  void dump(std::ostream &OS, size_t Indentation) const override {
    indent(OS, Indentation);
    OS << getName() << "\n";
    indent(OS, Indentation + 1);
    OS << "Containers\n";
    for (const auto &Name : getRunningContainersNames()) {
      indent(OS, Indentation + 2);
      OS << Name;
      OS << "\n";
    }
    if constexpr (Dumpable<PipeType>)
      ActualPipe.dump(OS, Indentation);
  }
};

} // namespace detail

/// This class is used to hide the unique ptr and expose a concrete class
/// instead of pointers, as well as implementing dump and operator=, which
/// is implemented as a clone.
class PipeWrapper {
private:
  std::unique_ptr<detail::PipeWrapperBase> Pipe;

public:
  template<typename PipeType>
  PipeWrapper(PipeType Pipe, std::vector<std::string> RunningContainersNames) :
    Pipe(makeUniqueWrapper(Pipe, std::move(RunningContainersNames))) {}

  PipeWrapper(const PipeWrapper &Other) : Pipe(Other.Pipe->clone()) {}
  PipeWrapper(PipeWrapper &&Other) = default;

  PipeWrapper &operator=(const PipeWrapper &Other);
  PipeWrapper &operator=(PipeWrapper &&Other) = default;

  ~PipeWrapper() = default;

public:
  template<typename PipeType>
  static PipeWrapper
  makeWrapper(std::vector<std::string> RunningContainersNames) {
    return PipeWrapper(PipeType(), std::move(RunningContainersNames));
  }

public:
  detail::PipeWrapperBase &operator*() { return *Pipe; }

  const detail::PipeWrapperBase &operator*() const { return *Pipe; }

  detail::PipeWrapperBase *operator->() { return Pipe.get(); }

  const detail::PipeWrapperBase *operator->() const { return Pipe.get(); }

public:
  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const {
    Pipe->dump(OS, Indentation + 1);
  }

  void dump() const { dump(dbg); }

private:
  template<typename PipeType>
  auto makeUniqueWrapper(PipeType Pipe,
                         std::vector<std::string> RunningContainersNames) {
    using Wrapper = detail::PipeWrapperImpl<PipeType>;
    return std::make_unique<Wrapper>(Pipe, std::move(RunningContainersNames));
  }
};

template<typename PipeType, typename... ContainerNames>
PipeWrapper bindPipe(ContainerNames &&...Names) {
  auto NamesList = { std::forward<ContainerNames>(Names)... };
  return PipeWrapper::makeWrapper<PipeType>(std::move(NamesList));
}

template<typename PipeType, typename... ContainerNames>
PipeWrapper bindPipe(PipeType &&E, ContainerNames &&...Names) {
  auto NamesList = { std::forward<ContainerNames>(Names)... };
  return PipeWrapper(std::forward<PipeType>(E), std::move(NamesList));
}

} // namespace pipeline
