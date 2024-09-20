#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Error.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/ContainerEnumerator.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/ExecutionContext.h"
#include "revng/Pipeline/Global.h"
#include "revng/Pipeline/Invokable.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Debug.h"

inline Logger<> InvalidationLog("invalidation");

namespace pipeline {

/// Represents the requested (not expected, which means that it contains only
/// the targets the user care about, not all those that will be generated as a
/// side effect) input and output of a given invocation of a pipe.
class PipeExecutionEntry {
public:
  ContainerToTargetsMap Output;
  ContainerToTargetsMap Input;

  PipeExecutionEntry(ContainerToTargetsMap Output,
                     ContainerToTargetsMap Input) :
    Output(std::move(Output)), Input(std::move(Input)) {}
};

namespace detail {

template<typename T>
concept HasContract = requires(T P) {
  { llvm::ArrayRef<ContractGroup>(P.getContract()) };
};

template<typename T, typename FirstRunArg, typename... Args>
concept Pipe = Invokable<T, FirstRunArg, Args...>
               and (IsContainer<Args> and ...) and HasContract<T>;

template<typename C, typename First, typename... Rest>
constexpr bool checkPipe(auto (C::*)(First, Rest...))
  requires Pipe<C, First, Rest...>
{
  return true;
}

template<typename C, typename First, typename... Rest>
constexpr size_t countArgs(auto (C::*)(First, Rest...)) {
  return sizeof...(Rest);
}

template<typename First, typename... Types>
const char *getNameOfContainerImpl(size_t Index) {
  if (Index == 0)
    return std::decay_t<First>::Name;

  if constexpr (sizeof...(Types) == 0)
    return "";
  else
    return getNameOfContainerImpl<Types...>(Index - 1);
}

template<typename C, typename Context, typename... Rest>
const char *getNameOfContainer(auto (C::*)(Context, Rest...), size_t Index) {
  return getNameOfContainerImpl<Rest...>(Index);
}

template<typename PipeType>
class PipeWrapperImpl;

// TODO: Rename, there are 3 layers of wrappers around a pipe and they are
// getting confusing
class PipeWrapperBase : public InvokableWrapperBase {
public:
  template<typename PipeType>
  using ImplType = PipeWrapperImpl<PipeType>;

public:
  virtual PipeExecutionEntry
  getRequirements(const Context &Context,
                  const ContainerToTargetsMap &Target) const = 0;

  virtual ContainerToTargetsMap
  deduceResults(const Context &Context,
                ContainerToTargetsMap &Target) const = 0;

  virtual bool areRequirementsMet(const Context &Context,
                                  const ContainerToTargetsMap &Input) const = 0;

  virtual std::unique_ptr<PipeWrapperBase>
  clone(std::vector<std::string> NewRunningContainersNames = {}) const = 0;

  virtual llvm::Error checkPrecondition(const Context &Context) const = 0;

  virtual size_t getContainerArgumentsCount() const = 0;

  virtual llvm::StringRef getContainerName(size_t Index) const = 0;

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
  bool areRequirementsMet(const Context &Context,
                          const ContainerToTargetsMap &Input) const override {
    const auto &Contracts = Invokable.getPipe().getContract();
    if (Contracts.size() == 0)
      return true;

    ContainerToTargetsMap ToCheck = Input;
    for (const auto &Contract : Contracts) {
      if (Contract.forwardMatches(Context,
                                  ToCheck,
                                  Invokable.getRunningContainersNames()))
        return true;

      Contract.deduceResults(Context,
                             ToCheck,
                             Invokable.getRunningContainersNames());
    }

    return false;
  }

  PipeExecutionEntry
  getRequirements(const Context &Context,
                  const ContainerToTargetsMap &Target) const override {
    revng_log(InvalidationLog,
              "Computing requirements for " << this->Invokable.getName());
    LoggerIndent<> Imdent(InvalidationLog);

    const auto &Contracts = Invokable.getPipe().getContract();
    ContainerToTargetsMap Input = Target;

    std::set<std::pair<std::string, unsigned>> ThisPipeOutputs;
    for (const auto &Contract : llvm::reverse(Contracts)) {
      Input = Contract
                .deduceRequirements(Context,
                                    Input,
                                    Invokable.getRunningContainersNames());

      for (const auto &[ContainerIndex, Kind] : Contract.getOutputs()) {
        auto RunningContainersNames = Invokable.getRunningContainersNames();
        if (not Invokable.isContainerArgumentConst(ContainerIndex)) {
          revng_log(InvalidationLog,
                    RunningContainersNames[ContainerIndex]
                      << " " << Kind->name().str() << " allowed");
          ThisPipeOutputs.insert({ RunningContainersNames[ContainerIndex],
                                   Kind->id() });
        }
      }
    }

    if (InvalidationLog.isEnabled()) {
      InvalidationLog << "Input:\n";
      Input.dump(InvalidationLog);

      InvalidationLog << DoLog;

      InvalidationLog << "Output (pre-filter):\n";
      Target.dump(InvalidationLog);
    }

    ContainerToTargetsMap Output = Target;
    // NOTE: do not iterate over the StringMap, the second entry of the pair is
    //       a copy (somehow)
    std::vector<std::string> Keys;
    for (llvm::StringRef Key : Output.keys())
      Keys.push_back(Key.str());

    for (const std::string &ContainerName : Keys) {
      auto &TargetList = Output.at(ContainerName);

      TargetList.erase_if([&](const class Target &T) -> bool {
        return not ThisPipeOutputs.contains({ ContainerName,
                                              T.getKind().id() });
      });

      if (TargetList.empty())
        Output.erase(ContainerName);
    }

    if (InvalidationLog.isEnabled()) {
      InvalidationLog << "Output:\n";
      Output.dump(InvalidationLog);
      InvalidationLog << DoLog;
    }

    return PipeExecutionEntry(Output, Input);
  }

  ContainerToTargetsMap
  deduceResults(const Context &Context,
                ContainerToTargetsMap &Target) const override {
    const auto &Contracts = Invokable.getPipe().getContract();

    for (const auto &Contract : Contracts) {
      Contract.deduceResults(Context,
                             Target,
                             Invokable.getRunningContainersNames());
    }

    return Target;
  }

  std::unique_ptr<PipeWrapperBase>
  clone(std::vector<std::string> NewContainersNames = {}) const override {
    if (NewContainersNames.empty())
      return std::make_unique<PipeWrapperImpl>(*this);
    return std::make_unique<PipeWrapperImpl>(*this,
                                             std::move(NewContainersNames));
  }

  llvm::Error checkPrecondition(const Context &Context) const override {
    return Invokable.getPipe().checkPrecondition(Context);
  }

  size_t getContainerArgumentsCount() const override {
    return countArgs(&PipeType::run);
  }

  llvm::StringRef getContainerName(size_t Index) const override {
    return getNameOfContainer(&PipeType::run, Index);
  }

public:
  void dump(std::ostream &OS,
            size_t Indentation) const override debug_function {
    Invokable.dump(OS, Indentation);
  }

  llvm::Error run(ExecutionContext &Context,
                  ContainerSet &Containers,
                  const llvm::StringMap<std::string> &ExtraArgs) override {
    return Invokable.run(Context, Containers, ExtraArgs);
  }

  void invalidate(const GlobalTupleTreeDiff &Diff,
                  ContainerToTargetsMap &Map,
                  const ContainerSet &Containers) const override {
    return Invokable.invalidate(Diff, Map, Containers);
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

// Due to invokable wrapper not being controllable by this file we need to have
// a extra wrapper that carries along the invalidation metadata too.
struct PipeWrapper {

  class InvalidationMetadata {
  private:
    llvm::StringMap<PathTargetBimap> PathCache;

  public:
    void registerTargetsDependingOn(const Context &Context,
                                    llvm::StringRef GlobalName,
                                    const TupleTreePath &Path,
                                    ContainerToTargetsMap &Out,
                                    Logger<> &Log) const {
      if (auto Iter = PathCache.find(GlobalName); Iter != PathCache.end()) {

        auto &Bimap = Iter->second;
        auto It = Bimap.find(Path);
        if (It == Bimap.end())
          return;

        if (Log.isEnabled()) {
          Log << "Registering: ";
          for (const auto &Entry : It->second) {
            Log << Entry.getTarget().serialize() << " in "
                << Entry.getContainerName() << "\n";
          }
          Log << DoLog;
        }

        for (const auto &Entry : It->second)
          Out.add(Entry.getContainerName(), Entry.getTarget());
      }
    }

    void remove(const ContainerToTargetsMap &Map) {
      for (auto &Pair : Map) {
        auto Iter = PathCache.find(Pair.first());
        if (Iter == PathCache.end())
          continue;

        Iter->second.remove(Pair.second, Pair.first());
      }
    }

    bool contains(llvm::StringRef GlobalName,
                  const TargetInContainer &Target) const {
      if (auto Iter = PathCache.find(GlobalName); Iter != PathCache.end())
        return Iter->second.contains(Target);
      return false;
    }

    const llvm::StringMap<PathTargetBimap> &getPathCache() const {
      return PathCache;
    }

    llvm::StringMap<PathTargetBimap> &getPathCache() { return PathCache; }

    const PathTargetBimap &getPathCache(llvm::StringRef GlobalName) const {
      revng_assert(PathCache.find(GlobalName) != PathCache.end());
      return PathCache.find(GlobalName)->second;
    }

    PathTargetBimap &getPathCache(llvm::StringRef GlobalName) {
      return PathCache[GlobalName];
    }

    void dump(const pipeline::Context &Context,
              unsigned Indentation = 0) const {
      for (const auto &[GlobalName, InvalidationData] : PathCache) {
        indent(dbg, Indentation);
        dbg << "Global " << GlobalName.str() << ":\n";

        for (const auto &[Path, Targets] : PathCache.find(GlobalName)->second) {
          indent(dbg, Indentation + 1);

          dbg << llvm::cantFail(Context.getGlobals().get(GlobalName))
                   ->serializePath(Path)
                   .value_or("(unavailable)")
              << ":\n";

          for (const TargetInContainer &Target : Targets) {
            Target.dump(dbg, Indentation + 2);
          }
        }
      }
    }
  };

public:
  using WrapperType = InvokableWrapper<detail::PipeWrapperBase>;
  WrapperType Pipe;
  InvalidationMetadata InvalidationMetadata;

public:
  template<typename PipeType>
  static PipeWrapper
  make(PipeType Pipe, std::vector<std::string> RunningContainersNames) {
    return WrapperType::make<PipeType>(Pipe, std::move(RunningContainersNames));
  }

  template<typename PipeType>
  static PipeWrapper make(std::vector<std::string> RunningContainersNames) {
    return WrapperType::make<PipeType>(std::move(RunningContainersNames));
  }

  PipeWrapper(const InvokableWrapper<detail::PipeWrapperBase> &Other) :
    Pipe(Other) {}

  PipeWrapper(const PipeWrapper &Other,
              std::vector<std::string> RunningContainersNames) :
    Pipe(Other.Pipe, RunningContainersNames) {}

  template<typename PipeType, typename... ContainerNames>
  static PipeWrapper bind(ContainerNames &&...Names) {
    return WrapperType::bind<PipeType, ContainerNames...>(Names...);
  }

  template<typename PipeType, typename... ContainerNames>
  static PipeWrapper bind(PipeType &&E, ContainerNames &&...Names) {
    return WrapperType::bind<PipeType, ContainerNames...>(E, Names...);
  }
};

} // namespace pipeline
