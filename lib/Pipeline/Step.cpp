/// \file Runner.cpp
/// A step is composed of a list of pipes and a set of containers representing
/// the content of the pipeline before the execution of such pipes.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Progress.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Errors.h"
#include "revng/Pipeline/Step.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

using namespace llvm;
using namespace std;
using namespace pipeline;

namespace pipeline {

class TargetInPipe {
public:
  std::string SerializedTarget;
  std::string PipeName;

  llvm::Expected<llvm::SmallVector<TargetInContainer, 2>>
  deserialize(const Context &Ctx, llvm::StringRef ContainerName) const;

  static TargetInPipe fromTargetInContainer(const TargetInContainer &Target,
                                            llvm::StringRef PipeName);
  bool operator<(const TargetInPipe &Other) const {
    const auto &Tied = std::tie(SerializedTarget, PipeName);
    return Tied < std::tie(Other.SerializedTarget, Other.PipeName);
  }
};

class ContainerInvalidationMetadata {
public:
  using ValueType = std::pair<pipeline::TargetInPipe, std::vector<std::string>>;
  using Vector = std::vector<ValueType>;
  Vector Data;

  void merge(ContainerInvalidationMetadata &&Other) {
    for (ValueType &Entry : Other.Data)
      Data.emplace_back(std::move(Entry));
  }

public:
  llvm::Expected<PathTargetBimap>
  deserialize(const Context &Ctx,
              const Global &Primitives,
              llvm::StringRef PipeName,
              llvm::StringRef ContainerName) const;

  static ContainerInvalidationMetadata serialize(const PathTargetBimap &Map,
                                                 const Global &Primitives,
                                                 llvm::StringRef PipeName,
                                                 llvm::StringRef ContainerName);
};

class NamedPathTargetBimapVector {
public:
  std::string GlobalName;
  ContainerInvalidationMetadata Map;
};
} // namespace pipeline

LLVM_YAML_IS_SEQUENCE_VECTOR(ContainerInvalidationMetadata::ValueType);

namespace llvm {
namespace yaml {

// YAML traits for TargetInContainer
template<>
struct MappingTraits<pipeline::TargetInPipe> {
  static void mapping(IO &IO, pipeline::TargetInPipe &TargetInContainer) {
    IO.mapRequired("Target", TargetInContainer.SerializedTarget);
    IO.mapRequired("PipeName", TargetInContainer.PipeName);
  }
};

template<>
struct MappingTraits<pipeline::ContainerInvalidationMetadata> {
  static void mapping(IO &Io,
                      pipeline::ContainerInvalidationMetadata &TargetMap) {
    Io.mapRequired("Map", TargetMap.Data);
  }
};

template<>
struct MappingTraits<ContainerInvalidationMetadata::Vector::value_type> {
  static void
  mapping(IO &Io,
          pipeline::ContainerInvalidationMetadata::Vector::value_type
            &TargetMap) {
    Io.mapRequired("Target", TargetMap.first.SerializedTarget);
    Io.mapRequired("PipeName", TargetMap.first.PipeName);
    Io.mapRequired("ReadPaths", TargetMap.second);
  }
};

} // namespace yaml
} // namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(pipeline::NamedPathTargetBimapVector);

namespace llvm {
namespace yaml {
template<>
struct MappingTraits<pipeline::NamedPathTargetBimapVector> {
  static void mapping(IO &Io, pipeline::NamedPathTargetBimapVector &TargetMap) {
    Io.mapRequired("GlobalName", TargetMap.GlobalName);
    Io.mapRequired("Map", TargetMap.Map.Data);
  }
};
} // namespace yaml
} // namespace llvm

llvm::Expected<llvm::SmallVector<TargetInContainer, 2>>
TargetInPipe::deserialize(const Context &Ctx,
                          llvm::StringRef ContainerName) const {
  TargetsList Targets;
  llvm::Error Error = parseTarget(Ctx,
                                  SerializedTarget,
                                  Ctx.getKindsRegistry(),
                                  Targets);
  if (Error)
    return std::move(Error);

  llvm::SmallVector<TargetInContainer, 2> Return;
  for (const Target &Target : Targets) {
    Return.emplace_back(std::move(Target), ContainerName.str());
  }
  return Return;
}

TargetInPipe
TargetInPipe::fromTargetInContainer(const TargetInContainer &Target,
                                    llvm::StringRef PipeName) {
  TargetInPipe ToReturn;
  ToReturn.PipeName = PipeName;
  ToReturn.SerializedTarget = Target.getTarget().serialize();

  return ToReturn;
}

ContainerInvalidationMetadata
ContainerInvalidationMetadata::serialize(const PathTargetBimap &Map,
                                         const Global &Global,
                                         llvm::StringRef PipeName,
                                         llvm::StringRef ContainerName) {
  ContainerInvalidationMetadata ToSerialize;
  std::map<pipeline::TargetInPipe, std::vector<std::string>> TemporaryMap;

  for (const auto &Content : Map) {
    for (const TargetInContainer &Entry : Content.second) {
      if (Entry.getContainerName() != ContainerName)
        continue;

      std::optional<std::string> AsString = Global.serializePath(Content.first);
      revng_check(AsString.has_value());
      TemporaryMap[TargetInPipe::fromTargetInContainer(Entry, PipeName)]
        .push_back(*AsString);
    }
  }

  for (const auto &Content : TemporaryMap) {
    std::pair ToEmplace{ Content.first, Content.second };
    ToSerialize.Data.emplace_back(std::move(ToEmplace));
  }

  return ToSerialize;
}

llvm::Expected<PathTargetBimap>
ContainerInvalidationMetadata::deserialize(const Context &Ctx,
                                           const Global &Global,
                                           llvm::StringRef PipeName,
                                           llvm::StringRef ContainerName)
  const {

  PathTargetBimap ToReturn;
  for (const ValueType &Entry : Data) {
    if (Entry.first.PipeName != PipeName) {
      continue;
    }

    llvm::Expected<SmallVector<TargetInContainer>>
      MaybeTarget = Entry.first.deserialize(Ctx, ContainerName);

    if (not MaybeTarget) {
      return MaybeTarget.takeError();
    }

    for (auto &SerializedPath : Entry.second) {
      std::optional<TupleTreePath>
        MaybeParsedPath = Global.deserializePath(SerializedPath);

      if (not MaybeParsedPath) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "could not parse " + SerializedPath);
      }

      for (const TargetInContainer &Path : *MaybeTarget) {
        ToReturn.insert(std::move(Path), std::move(*MaybeParsedPath));
      }
    }
  }

  return ToReturn;
}

std::pair<ContainerToTargetsMap, std::vector<PipeExecutionEntry>>
Step::analyzeGoals(const ContainerToTargetsMap &RequiredGoals) const {

  ContainerToTargetsMap AlreadyAvailable;
  ContainerToTargetsMap Targets = RequiredGoals;
  removeSatisfiedGoals(Targets, AlreadyAvailable);

  std::vector<PipeExecutionEntry> PipesExecutionEntries;
  for (const PipeWrapper &Pipe :
       llvm::make_range(Pipes.rbegin(), Pipes.rend())) {
    PipesExecutionEntries.push_back(Pipe.Pipe->getRequirements(*Ctx, Targets));
    Targets = PipesExecutionEntries.back().Input;
  }
  std::reverse(PipesExecutionEntries.begin(), PipesExecutionEntries.end());

  return std::make_pair(std::move(Targets), std::move(PipesExecutionEntries));
}

void Step::explainStartStep(const ContainerToTargetsMap &Targets,
                            size_t Indentation) const {

  indent(ExplanationLogger, Indentation);
  ExplanationLogger << "STARTING step on containers\n";
  indent(ExplanationLogger, Indentation + 1);
  ExplanationLogger << getName() << ":\n";
  prettyPrintStatus(Targets, ExplanationLogger, Indentation + 2);
  ExplanationLogger << DoLog;
}

void Step::explainEndStep(const ContainerToTargetsMap &Targets,
                          size_t Indentation) const {

  indent(ExplanationLogger, Indentation);
  ExplanationLogger << "ENDING step, the following have been produced\n";
  indent(ExplanationLogger, Indentation + 1);
  ExplanationLogger << getName() << ":\n";
  prettyPrintStatus(Targets, ExplanationLogger, Indentation + 2);
  ExplanationLogger << DoLog;
}

void Step::explainExecutedPipe(const InvokableWrapperBase &Wrapper,
                               size_t Indentation) const {
  ExplanationLogger << "RUN " << Wrapper.getName();
  ExplanationLogger << "(";

  std::vector<std::string> Vec = Wrapper.getRunningContainersNames();
  if (not Vec.empty()) {
    for (size_t I = 0; I < Vec.size() - 1; I++) {
      ExplanationLogger << Vec[I];
      ExplanationLogger << ", ";
    }
    ExplanationLogger << Vec.back();
  }

  ExplanationLogger << ")";
  ExplanationLogger << "\n";
  ExplanationLogger << DoLog;

  auto CommandStream = CommandLogger.getAsLLVMStream();
  Wrapper.print(*Ctx, *CommandStream, Indentation);
  CommandStream->flush();
  CommandLogger << DoLog;
}

ContainerSet Step::run(ContainerSet &&Input,
                       const std::vector<PipeExecutionEntry> &ExecutionInfos) {
  ContainerToTargetsMap InputEnumeration = Input.enumerate();
  explainStartStep(InputEnumeration);

  Task T(Pipes.size() + 1, "Step " + getName());
  for (const auto &[Pipe, Info] : llvm::zip(Pipes, ExecutionInfos)) {
    T.advance(Pipe.Pipe->getName(), false);
    explainExecutedPipe(*Pipe.Pipe);
    ExecutionContext Context(*Ctx, &Pipe, Info.Output);

    Pipe.Pipe->deduceResults(*Ctx, Context.getCurrentRequestedTargets());

    cantFail(Pipe.Pipe->run(Context, Input));
    llvm::cantFail(Input.verify());
  }

  T.advance("Merging back", true);
  explainEndStep(Input.enumerate());
  Containers.mergeBack(std::move(Input));
  InputEnumeration = deduceResults(InputEnumeration);
  ContainerSet Cloned = Containers.cloneFiltered(InputEnumeration);
  return Cloned;
}

void Step::pipeInvalidate(const GlobalTupleTreeDiff &Diff,
                          ContainerToTargetsMap &Map) const {
  for (const auto &Pipe : Pipes) {
    Pipe.Pipe->invalidate(Diff, Map, Containers);
  }
}

llvm::Error Step::runAnalysis(llvm::StringRef AnalysisName,
                              const ContainerToTargetsMap &Targets,
                              const llvm::StringMap<std::string> &ExtraArgs) {
  auto Stream = ExplanationLogger.getAsLLVMStream();
  ContainerToTargetsMap Map = Containers.enumerate();

  ContainerToTargetsMap CollapsedTargets = Targets;

  revng_assert(Map.contains(CollapsedTargets),
               "An analysis was requested, but not all targets are available");

  AnalysisWrapper &TheAnalysis = getAnalysis(AnalysisName);

  explainExecutedPipe(*TheAnalysis);

  ContainerSet Cloned = Containers.cloneFiltered(Targets);
  ExecutionContext ExecutionCtx(*Ctx, nullptr);
  return TheAnalysis->run(ExecutionCtx, Cloned, ExtraArgs);
}

void Step::removeSatisfiedGoals(TargetsList &RequiredInputs,
                                const ContainerBase &CachedSymbols,
                                TargetsList &ToLoad) {
  const TargetsList EnumeratedSymbols = CachedSymbols.enumerate();
  const auto IsCached = [&ToLoad,
                         &EnumeratedSymbols](const Target &Target) -> bool {
    bool MustBeLoaded = EnumeratedSymbols.contains(Target);
    if (MustBeLoaded)
      ToLoad.emplace_back(Target);
    return MustBeLoaded;
  };

  llvm::erase_if(RequiredInputs, IsCached);
}

void Step::removeSatisfiedGoals(ContainerToTargetsMap &Targets,
                                ContainerToTargetsMap &ToLoad) const {
  for (auto &RequiredInputsFromContainer : Targets) {
    llvm::StringRef ContainerName = RequiredInputsFromContainer.first();
    TargetsList &RequiredInputs = RequiredInputsFromContainer.second;
    TargetsList &ToLoadFromCurrentContainer = ToLoad[ContainerName];
    if (Containers.contains(ContainerName))
      removeSatisfiedGoals(RequiredInputs,
                           Containers.at(ContainerName),
                           ToLoadFromCurrentContainer);
  }
}

ContainerToTargetsMap Step::deduceResults(ContainerToTargetsMap Input) const {
  for (const PipeWrapper &Pipe : Pipes)
    Input = Pipe.Pipe->deduceResults(*Ctx, Input);
  return Input;
}

Error Step::invalidate(const ContainerToTargetsMap &ToRemove) {
  for (auto &Pipe : Pipes) {
    Pipe.InvalidationMetadata.remove(ToRemove);
  }
  return containers().remove(ToRemove);
}

Error Step::store(const revng::DirectoryPath &DirPath) const {
  if (auto Error = Containers.store(DirPath))
    return Error;

  return storeInvalidationMetadata(DirPath);
}

Error Step::checkPrecondition() const {
  for (const PipeWrapper &Pipe : Pipes) {
    if (llvm::Error Error = Pipe.Pipe->checkPrecondition(*Ctx); Error) {
      std::string Message = "While scheduling the " + Pipe.Pipe->getName()
                            + " pipe a precondition check failed:";
      return llvm::make_error<AnnotatedError>(std::move(Error), Message);
    }
  }
  return llvm::Error::success();
}

Error Step::load(const revng::DirectoryPath &DirPath) {
  auto MaybeBool = DirPath.exists();
  if (not MaybeBool)
    return MaybeBool.takeError();

  if (not MaybeBool.get())
    return llvm::Error::success();

  if (auto Error = Containers.load(DirPath))
    return Error;

  return loadInvalidationMetadata(DirPath);
}

llvm::Error
Step::loadInvalidationMetadataImpl(const revng::DirectoryPath &Path,
                                   ContainerSet::value_type &Container) {
  auto FilePath = Path.getFile(Container.first().str() + ".cache");
  auto MaybeBool = FilePath.exists();
  if (not MaybeBool)
    return MaybeBool.takeError();

  if (not MaybeBool.get())
    return llvm::Error::success();

  auto File = FilePath.getReadableFile();
  if (not File)
    return File.takeError();

  using Type = llvm::SmallVector<NamedPathTargetBimapVector, 2>;
  auto Parsed = ::deserialize<Type>(File.get()->buffer().getBuffer());
  if (not Parsed)
    return Parsed.takeError();

  for (PipeWrapper &Pipe : Pipes) {
    for (NamedPathTargetBimapVector &Entry : *Parsed) {
      Global *Global = llvm::cantFail(Ctx->getGlobals().get(Entry.GlobalName));
      auto Parsed(Entry.Map.deserialize(*Ctx,
                                        *Global,
                                        Pipe.Pipe->getName(),
                                        Container.first()));
      if (not Parsed)
        return Parsed.takeError();
      Pipe.InvalidationMetadata.getPathCache(Global->getName())
        .merge(std::move(*Parsed));
    }
  }
  return llvm::Error::success();
}

llvm::Error Step::loadInvalidationMetadata(const revng::DirectoryPath &Path) {

  for (PipeWrapper &Pipe : Pipes) {
    Pipe.InvalidationMetadata = {};
  }
  for (auto &Container : Containers) {
    if (llvm::Error Error = loadInvalidationMetadataImpl(Path, Container))
      return Error;
  }

  return llvm::Error::success();
}

llvm::Error
Step::storeInvalidationMetadata(const revng::DirectoryPath &Path) const {
  for (auto &Container : Containers) {
    if (Container.second == nullptr)
      continue;

    using Type = llvm::SmallVector<NamedPathTargetBimapVector, 2>;
    Type ToStore = {};

    for (const Global *Global : Ctx->getGlobals()) {
      NamedPathTargetBimapVector Entry;
      Entry.GlobalName = Global->getName();

      for (const PipeWrapper &Pipe : Pipes) {
        auto &PathCache = Pipe.InvalidationMetadata.getPathCache();
        if (PathCache.count(Entry.GlobalName) == 0)
          continue;

        using MetadataType = ContainerInvalidationMetadata;
        auto Serialize = MetadataType::serialize;
        MetadataType Serialized = Serialize(Pipe.InvalidationMetadata
                                              .getPathCache(Entry.GlobalName),
                                            *Global,
                                            Pipe.Pipe->getName(),
                                            Container.first());
        Entry.Map.merge(std::move(Serialized));
      }
      ToStore.emplace_back(std::move(Entry));
    }

    auto File = Path.getFile(Container.first().str() + ".cache")
                  .getWritableFile();
    if (not File)
      return File.takeError();
    ::serialize(File->get()->os(), ToStore);
    if (auto Error = File->get()->commit())
      return Error;
  }

  return llvm::Error::success();
}

std::vector<revng::FilePath>
Step::getWrittenFiles(const revng::DirectoryPath &DirPath) const {
  std::vector<revng::FilePath> Result = Containers.getWrittenFiles(DirPath);
  for (auto &Container : Containers)
    Result.push_back(DirPath.getFile(Container.first().str() + ".cache"));
  return Result;
}
