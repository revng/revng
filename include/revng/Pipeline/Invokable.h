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
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Pipeline/CLOption.h"
#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/ExecutionContext.h"
#include "revng/Pipeline/Option.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"

namespace pipeline {
template<typename T>
concept HasName = requires() {
  { T::Name } -> convertible_to<const char *>;
};

template<typename T>
concept HasInvalidate = requires() {
  { &T::invalidate };
};

template<typename T>
concept IsContainer = std::derived_from<std::decay_t<T>, ContainerBase>;

template<typename T>
concept IsNotContainer = not IsContainer<T>;

template<typename InvokableType, typename ReturnType, typename... Args>
constexpr ReturnType
invokableReturnTypeImpl(ReturnType (InvokableType::*F)(Args...)) {
  return ReturnType();
}

template<typename Type>
using invokableReturnType = decltype(invokableReturnTypeImpl(&Type::run));

template<typename Invokable>
constexpr bool invokableTypeReturnsError() {
  return std::is_same_v<invokableReturnType<Invokable>, llvm::Error>;
}

template<typename Invokable>
concept ReturnsError = invokableTypeReturnsError<Invokable>();

/// A Invokable is a class with the following characteristics:
///
/// * It must have a static constexpr field named Name that is a string
///   describing its name. Mostly used for debug purposes.
/// * a RetT run(T...) method where the first argument must be a
///   ExecutionContext& or const ExecutionContext&, arguments after the first
///   must be K& or const K& where K is derived from a container.
///
///   Options after the last container can be of any type, but for each of
///   them there must exists an entry in a constexpr tuple named Options that
///   defines their name and default value.
///
///   RetT can either be llvm::Error or void, if it is void then the invokable
///   never fails.
///
template<typename InvokableType, typename First, typename... Rest>
concept Invokable = convertible_to<ExecutionContext &, std::remove_cv_t<First>>
                    and HasName<InvokableType>;

namespace detail {
using StringArrayRef = llvm::ArrayRef<std::string>;

template<typename T>
auto &getContainer(ContainerSet &Containers, llvm::StringRef Name) {
  return llvm::cast<T>(Containers[Name]);
}

template<typename T>
auto &getContainer(const ContainerSet &Containers, llvm::StringRef Name) {
  return llvm::cast<T>(Containers.at(Name));
}

template<typename T, size_t I>
  requires PipelineOptionType<OptionType<T, I>>
OptionType<T, I> deserializeImpl(llvm::StringRef Value) {
  using ReturnType = OptionType<T, I>;

  if constexpr (std::is_same_v<std::string, ReturnType>) {
    return Value.str();
  } else {
    ReturnType Result;
    revng_assert(not Value.consumeInteger(10, Result));
    return Result;
  }
}

template<typename T, size_t I>
OptionType<T, I> getOption(const llvm::StringMap<std::string> &Map) {
  using OptionT = OptionType<T, I>;
  using llvm::StringRef;
  std::string Name = (StringRef(T::Name) + "-" + getOptionName<T, I>()).str();

  if (auto Iter = Map.find(Name); Iter != Map.end()) {
    return deserializeImpl<T, I>(Iter->second);
  }

  // Handle --$OPTION-path flag
  if constexpr (std::is_same_v<OptionT, std::string>) {
    auto &PathOpt = CLOptionBase::getOption<std::string>(Name + "-path");
    if (not PathOpt.get().empty()) {
      using llvm::MemoryBuffer;
      auto MaybeBuffer = MemoryBuffer::getFile(PathOpt.get());
      revng_assert(MaybeBuffer);
      return MaybeBuffer->get()->getBuffer().str();
    }
  }

  // Handle --$OPTION flag
  if (auto &Option = CLOptionBase::getOption<OptionT>(Name); Option.isSet()) {
    return Option.get();
  }

  // Use default
  return getOptionDefault<T, I>();
}

template<typename DeducedContextType,
         typename InvokableType,
         typename... AllArgs,
         IsContainer... Args,
         IsNotContainer... OptionArgsTypes,
         size_t... S,
         size_t... OptionArgsIndexes>
auto invokeImpl(DeducedContextType &Ctx,
                InvokableType &Pipe,
                auto (InvokableType::*F)(DeducedContextType &, AllArgs...),
                ContainerSet &Containers,
                const StringArrayRef &ArgsNames,
                const std::tuple<Args...> *,
                const std::tuple<OptionArgsTypes...> *,
                const std::integer_sequence<size_t, S...> &,
                const llvm::StringMap<std::string> &OptionArgs,
                const std::integer_sequence<size_t, OptionArgsIndexes...> &) {

  using namespace std;
  return (Pipe.*F)(Ctx,
                   getContainer<decay_t<Args>>(Containers, ArgsNames[S])...,
                   getOption<InvokableType, OptionArgsIndexes>(OptionArgs)...);
}

template<typename InvokableType,
         typename... AllArgs,
         IsContainer... Args,
         size_t... S>
auto invokeInvalidateImpl(const InvokableType &Pipe,
                          auto (InvokableType::*F)(AllArgs...) const,
                          const ContainerSet &Containers,
                          const StringArrayRef &ArgsNames,
                          const std::tuple<Args...> *,
                          const std::integer_sequence<size_t, S...> &,
                          const GlobalTupleTreeDiff &Diff) {

  using namespace std;
  return (Pipe.*F)(getContainer<decay_t<Args>>(Containers, ArgsNames[S])...,
                   Diff);
}

template<typename... T>
using Tuple = std::tuple<T...>;

template<bool B, typename T, typename F>
using Cond_t = std::conditional_t<B, T, F>;

template<typename T>
using ContainerToTuple = Cond_t<IsContainer<T>, Tuple<T>, Tuple<>>;

template<typename T>
using NonContainerTypeToTuple = Cond_t<IsContainer<T>, Tuple<>, Tuple<T>>;

template<typename... Args, typename... Args2>
std::tuple<Args..., Args2...> *
concatImpl(const std::tuple<Args...> *, const std::tuple<Args2...> *) {
  return nullptr;
}

template<typename T1, typename T2>
using ConcatImpl = std::decay_t<decltype(*concatImpl(std::declval<T1 *>(),
                                                     std::declval<T2 *>()))>;

template<typename First, typename... Rest>
auto *concatMultiple(const First *, const Rest *...Others) {
  if constexpr (sizeof...(Rest) == 0) {
    return static_cast<First *>(nullptr);
  } else {
    auto *Recurred = concatMultiple<Rest...>(Others...);
    using TupleType = std::decay_t<decltype(*Recurred)>;
    using Concatted = ConcatImpl<First, TupleType>;
    return static_cast<Concatted *>(nullptr);
  }
}

inline std::tuple<> *concatMultiple() {
  return nullptr;
}

template<typename T>
using Decay = std::decay_t<T>;

template<typename... T>
using TupleConcat = Decay<decltype(*concatMultiple(std::declval<T *>()...))>;

template<typename... T>
using FilterContainers = TupleConcat<ContainerToTuple<T>...>;

template<typename... T>
using FilterNonContainers = TupleConcat<NonContainerTypeToTuple<T>...>;

template<typename T, size_t S, typename... ArgsT>
void createAndAppend(std::vector<std::unique_ptr<CLOptionBase>> &Out,
                     ArgsT &&...Args) {
  using Wrapper = CLOptionWrapper<OptionType<T, S>>;
  Out.emplace_back(std::make_unique<Wrapper>(std::forward<ArgsT>(Args)...));
}

template<typename T, size_t S, typename... ArgsT>
void createAndAppendPathSwitch(std::vector<std::unique_ptr<CLOptionBase>> &Out,
                               ArgsT &&...Args) {
  if constexpr (std::is_same_v<OptionType<T, S>, std::string>) {
    using Wrapper = CLOptionWrapper<std::string>;
    Out.emplace_back(std::make_unique<Wrapper>(std::forward<ArgsT>(Args)...));
  }
}

template<typename T, size_t... S>
void createCLOption(std::vector<std::unique_ptr<CLOptionBase>> &Out,
                    const std::integer_sequence<size_t, S...> &,
                    llvm::cl::OptionCategory *Cat = nullptr) {
  using cat = llvm::cl::cat;
  (createAndAppend<T, S>(Out, T::Name, getOptionName<T, S>(), cat(*Cat)), ...);
  (createAndAppendPathSwitch<T, S>(Out,
                                   T::Name,
                                   getOptionName<T, S>().str() + "-path",
                                   cat(*Cat)),
   ...);
}

template<typename T, typename DeducedContextType, typename... AllArgs>
std::vector<std::unique_ptr<CLOptionBase>>
createCLOptionsImpl(auto (T::*F)(DeducedContextType &, AllArgs...),
                    llvm::cl::OptionCategory *Category = nullptr) {
  using OptionArgsTypes = detail::FilterNonContainers<AllArgs...>;
  constexpr size_t OptionArgsCount = std::tuple_size<OptionArgsTypes>::value;
  constexpr auto
    OptionArgsIndexes = std::make_integer_sequence<size_t, OptionArgsCount>();

  std::vector<std::unique_ptr<CLOptionBase>> Out;
  createCLOption<T>(Out, OptionArgsIndexes, Category);
  return Out;
}

template<typename T, size_t... S>
void getOptionNamesFromIndexes(std::vector<std::string> &Out,
                               const std::integer_sequence<size_t, S...> &) {
  (Out.push_back(getOptionName<T, S>().str()), ...);
}

template<typename T, typename CtxT, typename... Args>
std::vector<std::string> getOptionsNamesImpl(auto (T::*F)(CtxT &, Args...)) {

  using OptionArgsTypes = detail::FilterNonContainers<Args...>;
  constexpr size_t OptionArgsCount = std::tuple_size<OptionArgsTypes>::value;
  constexpr auto
    OptionArgsIndexes = std::make_integer_sequence<size_t, OptionArgsCount>();

  std::vector<std::string> Out;
  getOptionNamesFromIndexes<T>(Out, OptionArgsIndexes);

  return Out;
}

template<typename T>
std::vector<std::string> getOptionsNames() {
  return getOptionsNamesImpl<T>(&T::run);
}

template<typename T, size_t... S>
void getOptionTypeFromIndexes(std::vector<std::string> &Out,
                              const std::integer_sequence<size_t, S...> &) {
  (Out.push_back(getTypeName<T, S>().str()), ...);
}

template<typename T, typename CtxT, typename... Args>
std::vector<std::string> getOptionsTypesImpl(auto (T::*F)(CtxT &, Args...)) {

  using OptionArgsTypes = detail::FilterNonContainers<Args...>;
  constexpr size_t OptionArgsCount = std::tuple_size<OptionArgsTypes>::value;
  constexpr auto
    OptionArgsIndexes = std::make_integer_sequence<size_t, OptionArgsCount>();

  std::vector<std::string> Out;
  getOptionTypeFromIndexes<T>(Out, OptionArgsIndexes);

  return Out;
}

template<typename T>
std::vector<std::string> getOptionsTypes() {
  return getOptionsTypesImpl<T>(&T::run);
}

template<typename First, typename... Rest>
constexpr bool isNthTypeConst(size_t I) {
  if (I == 0)
    return std::is_const_v<std::remove_reference_t<First>>;

  if constexpr (sizeof...(Rest) == 0)
    return false;
  else
    return isNthTypeConst<Rest...>(I - 1);
}

template<typename InvokableType, typename... Args>
constexpr bool
isRunArgumentConstImpl(auto (InvokableType::*F)(Args...), size_t Index) {
  return isNthTypeConst<Args...>(Index);
}

template<typename InvokableType>
constexpr bool isRunArgumentConst(size_t ArgumentIndex) {
  return isRunArgumentConstImpl(&InvokableType::run, ArgumentIndex);
}
} // namespace detail

template<typename T>
std::vector<std::unique_ptr<CLOptionBase>>
createCLOptions(llvm::cl::OptionCategory *Category = nullptr) {
  return detail::createCLOptionsImpl<T>(&T::run, Category);
}

/// Invokes the F member function on the Pipe Pipe passing as nth argument the
/// container with the name equal to the nth element of ArgsNames.
template<typename InvokableType, typename ContextT, typename... Args>
auto invokePipeFunction(ExecutionContext &Ctx,
                        InvokableType &Pipe,
                        auto (InvokableType::*F)(ContextT &, Args...),
                        ContainerSet &Containers,
                        const llvm::ArrayRef<std::string> &ArgsNames,
                        const llvm::StringMap<std::string> &OptionArgs) {

  using ContainersTypes = detail::FilterContainers<Args...>;
  constexpr size_t ContainersCount = std::tuple_size<ContainersTypes>::value;
  constexpr auto
    Indexes = std::make_integer_sequence<size_t, ContainersCount>();
  revng_assert(ContainersCount == ArgsNames.size());

  using OptionArgsTypes = detail::FilterNonContainers<Args...>;
  constexpr size_t OptionArgsCount = std::tuple_size<OptionArgsTypes>::value;
  constexpr auto
    OptionArgsIndexes = std::make_integer_sequence<size_t, OptionArgsCount>();

  return detail::invokeImpl<ContextT>(Ctx,
                                      Pipe,
                                      F,
                                      Containers,
                                      ArgsNames,
                                      static_cast<ContainersTypes *>(nullptr),
                                      static_cast<OptionArgsTypes *>(nullptr),
                                      Indexes,
                                      OptionArgs,
                                      OptionArgsIndexes);
}

/// Invokes the F member function on the Pipe Pipe passing as nth argument the
/// container with the name equal to the nth element of ArgsNames.
template<typename InvokableType, typename... Args>
auto invokeInvalidateFunction(const InvokableType &Pipe,
                              auto (InvokableType::*F)(Args...) const,
                              const ContainerSet &Containers,
                              const llvm::ArrayRef<std::string> &ArgsNames,
                              const GlobalTupleTreeDiff &Diff) {

  using ContainersTypes = detail::FilterContainers<Args...>;
  constexpr size_t ContainersCount = std::tuple_size<ContainersTypes>::value;
  constexpr auto
    Indexes = std::make_integer_sequence<size_t, ContainersCount>();
  revng_assert(ContainersCount == ArgsNames.size());

  return detail::invokeInvalidateImpl(Pipe,
                                      F,
                                      Containers,
                                      ArgsNames,
                                      static_cast<ContainersTypes *>(nullptr),
                                      Indexes,
                                      Diff);
}

template<typename T>
concept Dumpable = requires(T D) {
  { D.dump(dbg, 0) };
};

template<typename InvokableType>
concept Printable = requires(InvokableType Pipe) {
  {
    Pipe.print(std::declval<const Context &>(),
               llvm::outs(),
               std::declval<llvm::ArrayRef<std::string>>())
  };
};

class InvokableWrapperBase {
public:
  virtual llvm::Error run(ExecutionContext &Ctx,
                          ContainerSet &Containers,
                          const llvm::StringMap<std::string> &Options = {}) = 0;

  virtual void invalidate(const GlobalTupleTreeDiff &Diff,
                          ContainerToTargetsMap &Map,
                          const ContainerSet &Containers) const = 0;

  virtual ~InvokableWrapperBase() = default;
  virtual std::vector<std::string> getRunningContainersNames() const = 0;
  virtual std::string getName() const = 0;
  virtual void dump(std::ostream &OS, size_t Indents) const = 0;
  virtual bool isContainerArgumentConst(size_t ArgumentIndex) const = 0;
  virtual std::vector<std::string> getOptionsNames() const = 0;
  virtual std::vector<std::string> getOptionsTypes() const = 0;
};

/// A pipe must be type erased somehow to become compatible with a pipeline,
/// a PipeWrapperImpl takes care of this issue, it can be constructed from
/// any pipeline type, and it will expose the contract and run method of that
/// enforcer.
template<typename InvokableType>
class InvokableWrapperImpl : public InvokableWrapperBase {
private:
  InvokableType ActualPipe;
  std::vector<std::string> RunningContainersNames;

public:
  InvokableWrapperImpl(InvokableType ActualPipe,
                       std::vector<std::string> RunningContainersNames) :
    ActualPipe(std::move(ActualPipe)),
    RunningContainersNames(std::move(RunningContainersNames)) {}

  InvokableWrapperImpl(const InvokableWrapperImpl &Other,
                       std::vector<std::string> RunningContainersNames) :
    ActualPipe(Other.ActualPipe),
    RunningContainersNames(std::move(RunningContainersNames)) {}

  ~InvokableWrapperImpl() override = default;

public:
  std::string getName() const override { return InvokableType::Name; }

public:
  llvm::Error run(ExecutionContext &Ctx,
                  ContainerSet &Containers,
                  const llvm::StringMap<std::string> &OptionArgs) override {
    if constexpr (invokableTypeReturnsError<InvokableType>()) {
      return invokePipeFunction(Ctx,
                                ActualPipe,
                                &InvokableType::run,
                                Containers,
                                RunningContainersNames,
                                OptionArgs);
    } else {
      invokePipeFunction(Ctx,
                         ActualPipe,
                         &InvokableType::run,
                         Containers,
                         RunningContainersNames,
                         OptionArgs);
    }
    return llvm::Error::success();
  }

  void invalidate(const GlobalTupleTreeDiff &Diff,
                  ContainerToTargetsMap &Map,
                  const ContainerSet &Containers) const override {
    if constexpr (HasInvalidate<InvokableType>) {
      std::map<const ContainerBase *, TargetsList> Result;

      auto IsAvailable = [&Containers](const std::string &Name) -> bool {
        return Containers.contains(Name);
      };

      if (llvm::all_of(RunningContainersNames, IsAvailable)) {
        Result = invokeInvalidateFunction(ActualPipe,
                                          &InvokableType::invalidate,
                                          Containers,
                                          RunningContainersNames,
                                          Diff);
      }

      for (auto &[Container, Targets] : Result) {
        Map[Container->name()].merge(Targets);
      }
    }
  }

public:
  std::vector<std::string> getRunningContainersNames() const override {
    return RunningContainersNames;
  }

  const InvokableType &getPipe() const { return ActualPipe; }

  std::vector<std::string> getOptionsNames() const override {
    return detail::getOptionsNames<InvokableType>();
  }

  std::vector<std::string> getOptionsTypes() const override {
    return detail::getOptionsTypes<InvokableType>();
  }

public:
  void dump(std::ostream &OS, size_t Indentation) const override {
    indent(OS, Indentation);
    OS << getName() << "\n";
    indent(OS, Indentation + 1);
    if (const auto &Names = getRunningContainersNames(); !Names.empty()) {
      OS << "Containers:\n";
      for (const auto &Name : Names) {
        indent(OS, Indentation + 2);
        OS << Name;
        OS << "\n";
      }
    } else {
      OS << "No containers.\n";
    }
    if constexpr (Dumpable<InvokableType>)
      ActualPipe.dump(OS, Indentation);
  }

  bool isContainerArgumentConst(size_t ArgumentIndex) const override {
    return detail::isRunArgumentConst<InvokableType>(ArgumentIndex + 1);
  }
};

/// This class is used to hide the unique ptr and expose a concrete class
/// instead of pointers, as well as implementing dump and operator=, which
/// is implemented as a clone.
template<typename BaseInterfaceT>
class InvokableWrapper {
public:
  template<typename T>
  using WrapperT = typename BaseInterfaceT::template ImplType<T>;
  using BaseInterface = BaseInterfaceT;

private:
  std::unique_ptr<BaseInterface> Pipe;

public:
  InvokableWrapper(const InvokableWrapper &Other,
                   std::vector<std::string> RunningContainersNames) :
    Pipe(Other.Pipe->clone(std::move(RunningContainersNames))) {}
  InvokableWrapper(const InvokableWrapper &Other) : Pipe(Other.Pipe->clone()) {}
  InvokableWrapper(InvokableWrapper &&Other) = default;

  InvokableWrapper &operator=(const InvokableWrapper &Other) {
    if (this == &Other)
      return *this;

    Pipe = Other.Pipe->clone();
    return *this;
  }

  InvokableWrapper &operator=(InvokableWrapper &&Other) = default;

  ~InvokableWrapper() = default;

public:
  template<typename PipeType>
  static InvokableWrapper
  make(std::vector<std::string> RunningContainersNames) {
    using Wrapper = WrapperT<PipeType>;
    auto Ptr = std::make_unique<Wrapper>(PipeType(),
                                         std::move(RunningContainersNames));
    return InvokableWrapper(std::move(Ptr));
  }

  template<typename PipeType>
  static InvokableWrapper
  make(PipeType Pipe, std::vector<std::string> RunningContainersNames) {
    using Wrapper = WrapperT<PipeType>;
    auto Ptr = std::make_unique<Wrapper>(std::move(Pipe),
                                         std::move(RunningContainersNames));
    return InvokableWrapper(std::move(Ptr));
  }

  template<typename PipeType, typename... ContainerNames>
  static InvokableWrapper bind(ContainerNames &&...Names) {
    auto NamesList = { std::forward<ContainerNames>(Names)... };
    return make<PipeType>(std::move(NamesList));
  }

  template<typename PipeType, typename... ContainerNames>
  static InvokableWrapper bind(PipeType &&E, ContainerNames &&...Names) {
    auto NamesList = { std::forward<ContainerNames>(Names)... };
    return make(std::forward<PipeType>(E), std::move(NamesList));
  }

public:
  BaseInterface &operator*() { return *Pipe; }

  const BaseInterface &operator*() const { return *Pipe; }

  BaseInterface *operator->() { return Pipe.get(); }

  const BaseInterface *operator->() const { return Pipe.get(); }

public:
  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const {
    Pipe->dump(OS, Indentation + 1);
  }

  void dump() const { dump(dbg); }

private:
  InvokableWrapper(std::unique_ptr<BaseInterfaceT> Pipe) :
    Pipe(std::move(Pipe)) {}
};
} // namespace pipeline
