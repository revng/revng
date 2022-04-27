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
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
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

/// A Invokable is a class with the following characteristics:
///
/// * It must have a static constexpr field named Name that is a string
///   describing its name. Mostly used for debug purposes.
/// * a void run(T...) method where the first argument must be a Context and
///   every other type must be the most derived type of a Container.
template<typename InvokableType, typename FirstRunArg, typename... Rest>
concept Invokable = HasName<InvokableType> and(IsContainer<Rest> and...)
                    and (IsContext<FirstRunArg> || IsConstContext<FirstRunArg>);

namespace detail {
using StringArrayRef = llvm::ArrayRef<std::string>;

template<typename T>
auto &getContainerFromName(ContainerSet &Containers, llvm::StringRef Name) {
  return llvm::cast<T>(Containers[Name]);
}

template<typename InvokableType, typename... Args, size_t... S>
void invokeImpl(Context &Ctx,
                InvokableType &Pipe,
                void (InvokableType::*F)(Context &, Args...),
                ContainerSet &Containers,
                const StringArrayRef &ArgsNames,
                const std::integer_sequence<size_t, S...> &) {

  using namespace std;
  (Pipe.*F)(Ctx,
            getContainerFromName<decay_t<Args>>(Containers, ArgsNames[S])...);
}

template<typename InvokableType, typename... Args, size_t... S>
void invokeImpl(const Context &Ctx,
                InvokableType &Pipe,
                void (InvokableType::*F)(const Context &, Args...),
                ContainerSet &Containers,
                const StringArrayRef &ArgsNames,
                const std::integer_sequence<size_t, S...> &) {
  using namespace std;
  (Pipe.*F)(Ctx,
            getContainerFromName<decay_t<Args>>(Containers, ArgsNames[S])...);
}
} // namespace detail

/// Invokes the F member function on the Pipe Pipe passing as nth argument the
/// container with the name equal to the nth element of ArgsNames.
template<typename InvokableType, typename... Args>
void invokePipeFunction(Context &Ctx,
                        InvokableType &Pipe,
                        void (InvokableType::*F)(Context &, Args...),
                        ContainerSet &Containers,
                        const llvm::ArrayRef<std::string> &ArgsNames) {
  constexpr auto
    Indexes = std::make_integer_sequence<size_t, sizeof...(Args)>();
  revng_assert(sizeof...(Args) == ArgsNames.size());
  detail::invokeImpl(Ctx, Pipe, F, Containers, ArgsNames, Indexes);
}

template<typename InvokableType, typename... Args>
void invokePipeFunction(Context &Ctx,
                        InvokableType &Pipe,
                        void (InvokableType::*F)(const Context &, Args...),
                        ContainerSet &Containers,
                        const llvm::ArrayRef<std::string> &ArgsNames) {
  constexpr auto
    Indexes = std::make_integer_sequence<size_t, sizeof...(Args)>();
  revng_assert(sizeof...(Args) == ArgsNames.size());
  detail::invokeImpl(Ctx, Pipe, F, Containers, ArgsNames, Indexes);
}

template<typename T>
concept Dumpable = requires(T D) {
  { D.dump(dbg, 0) };
};

template<typename InvokableType>
concept Printable = requires(InvokableType Pipe) {
  { Pipe.print(std::declval<const Context &>(),
               llvm::outs(),
               std::declval<llvm::ArrayRef<std::string>>()) };
};

class InvokableWrapperBase {
public:
  virtual void run(Context &Ctx, ContainerSet &Containers) = 0;
  virtual ~InvokableWrapperBase() = default;
  virtual std::vector<std::string> getRunningContainersNames() const = 0;
  virtual std::string getName() const = 0;
  virtual void dump(std::ostream &OS, size_t Indents) const = 0;
  virtual void
  print(const Context &Ctx, llvm::raw_ostream &OS, size_t Indents) const = 0;
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

  ~InvokableWrapperImpl() override = default;

public:
  std::string getName() const override { return InvokableType::Name; }

public:
  void run(Context &Ctx, ContainerSet &Containers) override {
    invokePipeFunction(Ctx,
                       ActualPipe,
                       &InvokableType::run,
                       Containers,
                       RunningContainersNames);
  }

public:
  std::vector<std::string> getRunningContainersNames() const override {
    return RunningContainersNames;
  }

  const InvokableType &getPipe() const { return ActualPipe; }

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
    if constexpr (Dumpable<InvokableType>)
      ActualPipe.dump(OS, Indentation);
  }

  void print(const Context &Ctx,
             llvm::raw_ostream &OS,
             size_t Indentation) const override {
    if constexpr (Printable<InvokableType>) {
      indent(OS, Indentation);
      const auto &Names = getRunningContainersNames();
      ActualPipe.print(Ctx, OS, Names);
    }
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
