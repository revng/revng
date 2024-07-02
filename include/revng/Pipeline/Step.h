#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>
#include <optional>
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
#include "llvm/Support/raw_ostream.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Pipeline/Analysis.h"
#include "revng/Pipeline/ContainerSet.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Pipe.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

namespace pipeline {

/// A step is a list of pipes that must be executed entirely or not at all.
/// Furthermore a step has a set of containers associated to it as well that
/// will contain the element used for perform the computations.
class Step {
public:
  // notice we need iterator stability of analyses because analyses list refers
  // to them
  using AnalysisMapType = llvm::StringMap<AnalysisWrapper>;
  using AnalysisIterator = AnalysisMapType::iterator;
  using AnalysisValueType = AnalysisMapType::value_type;
  using ConstAnalysisIterator = AnalysisMapType::const_iterator;

private:
  struct ArtifactsInfo {
    std::string Container;
    const Kind *Kind;
    std::string SingleTargetFilename;

    ArtifactsInfo() : Container(), Kind(nullptr), SingleTargetFilename() {}
    ArtifactsInfo(std::string Container,
                  const pipeline::Kind *Kind,
                  std::string SingleTargetFilename) :
      Container(std::move(Container)),
      Kind(Kind),
      SingleTargetFilename(std::move(SingleTargetFilename)) {}

    bool isValid() const {
      return !Container.empty() && Kind != nullptr
             && !SingleTargetFilename.empty();
    }
  };

  std::string Name;
  std::string Component;
  ContainerSet Containers;
  std::vector<PipeWrapper> Pipes;
  Step *PreviousStep;
  ArtifactsInfo Artifacts;
  AnalysisMapType AnalysisMap;
  Context *Ctx;

public:
  template<typename... PipeWrapperTypes>
  Step(Context &Ctx,
       std::string Name,
       std::string Component,
       ContainerSet Containers,
       PipeWrapperTypes &&...PipeWrappers) :
    Name(std::move(Name)),
    Component(std::move(Component)),
    Containers(std::move(Containers)),
    Pipes({ std::forward<PipeWrapperTypes>(PipeWrappers)... }),
    PreviousStep(nullptr),
    Ctx(&Ctx) {}

  template<typename... PipeWrapperTypes>
  Step(Context &Ctx,
       std::string Name,
       std::string Component,
       ContainerSet Containers,
       Step &PreviousStep,
       PipeWrapperTypes &&...PipeWrappers) :
    Name(std::move(Name)),
    Component(std::move(Component)),
    Containers(std::move(Containers)),
    Pipes({ std::forward<PipeWrapperTypes>(PipeWrappers)... }),
    PreviousStep(&PreviousStep),
    Ctx(&Ctx) {}

public:
  // TODO: drop the Out parameter pattern if favour of coroutines in the whole
  // codebase.
  void registerTargetsDependingOn(llvm::StringRef GlobalName,
                                  const TupleTreePath &Path,
                                  TargetInStepSet &Out) const {
    ContainerToTargetsMap OutMap;
    for (const PipeWrapper &Pipe : Pipes) {
      Pipe.InvalidationMetadata.registerTargetsDependingOn(*Ctx,
                                                           GlobalName,
                                                           Path,
                                                           OutMap);
    }
    for (auto &Container : OutMap) {
      if (Containers.contains(Container.first()))
        Container.second = Container.second.intersect(Containers
                                                        .at(Container.first())
                                                        .enumerate());
    }

    Out[getName()].merge(OutMap);
  }

  bool invalidationMetadataContains(llvm::StringRef GlobalName,
                                    const TargetInContainer &Target) const {
    for (const PipeWrapper &Pipe : Pipes) {
      if (Pipe.InvalidationMetadata.contains(GlobalName, Target))
        return true;
    }
    return false;
  }

private:
  llvm::Error loadInvalidationMetadataImpl(const revng::DirectoryPath &Path,
                                           ContainerSet::value_type &Pair);

private:
  llvm::Error loadInvalidationMetadata(const revng::DirectoryPath &Path);

  llvm::Error storeInvalidationMetadata(const revng::DirectoryPath &Path) const;

public:
  void addAnalysis(llvm::StringRef Name, AnalysisWrapper Analysis) {
    AnalysisMap.try_emplace(Name, std::move(Analysis));
  }

  const AnalysisWrapper &getAnalysis(llvm::StringRef Name) const {
    return AnalysisMap.find(Name)->second;
  }

  AnalysisWrapper &getAnalysis(llvm::StringRef Name) {
    return AnalysisMap.find(Name)->second;
  }

  bool hasAnalysis(llvm::StringRef Name) const {
    return AnalysisMap.find(Name) != AnalysisMap.end();
  }

  AnalysisIterator analysesBegin() { return AnalysisMap.begin(); }
  AnalysisIterator analysesEnd() { return AnalysisMap.end(); }
  ConstAnalysisIterator analysesBegin() const { return AnalysisMap.begin(); }
  ConstAnalysisIterator analysesEnd() const { return AnalysisMap.end(); }
  llvm::iterator_range<AnalysisIterator> analyses() {
    return llvm::make_range(AnalysisMap.begin(), AnalysisMap.end());
  }
  llvm::iterator_range<ConstAnalysisIterator> analyses() const {
    return llvm::make_range(AnalysisMap.begin(), AnalysisMap.end());
  }

  size_t getAnalysesSize() const { return AnalysisMap.size(); }

public:
  llvm::StringRef getName() const { return Name; }
  const ContainerSet &containers() const { return Containers; }
  ContainerSet &containers() { return Containers; }

  std::set<llvm::StringRef> mutableContainers() const {
    std::set<llvm::StringRef> MutableContainers;
    for (const auto &Pipe : Pipes) {
      size_t ArgumentsCount = Pipe.Pipe->getContainerArgumentsCount();
      for (size_t I = 0; I < ArgumentsCount; ++I) {
        if (not Pipe.Pipe->isContainerArgumentConst(I)) {
          MutableContainers.insert(Pipe.Pipe->getRunningContainersNames()[I]);
        }
      }
    }

    return MutableContainers;
  }

  llvm::Error setArtifacts(std::string ContainerName,
                           const Kind *ArtifactsKind,
                           std::string SingleTargetFilename) {
    if (Containers.contains(ContainerName)) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Artifact Container does not exist");
    }
    Artifacts = ArtifactsInfo(std::move(ContainerName),
                              ArtifactsKind,
                              std::move(SingleTargetFilename));
    return llvm::Error::success();
  }

  llvm::StringRef getComponent() const { return Component; }

  const Kind *getArtifactsKind() const {
    if (Artifacts.isValid()) {
      return Artifacts.Kind;
    } else {
      return nullptr;
    }
  }

  std::string getArtifactsContainerName() const {
    if (!Artifacts.isValid())
      return "";
    else
      return Artifacts.Container;
  }

  const ContainerSet::value_type *getArtifactsContainer() {

    if (!Artifacts.isValid()) {
      return nullptr;
    }

    const std::string &ContainerName = Artifacts.Container;
    if (Containers.isContainerRegistered(ContainerName)) {
      Containers[ContainerName];
      return &*Containers.find(ContainerName);
    } else {
      return nullptr;
    }
  }

  llvm::StringRef getArtifactsSingleTargetFilename() const {
    if (!Artifacts.isValid()) {
      return llvm::StringRef();
    }

    return Artifacts.SingleTargetFilename;
  }

public:
  bool hasPredecessor() const { return PreviousStep != nullptr; }

  const Step &getPredecessor() const {
    revng_assert(PreviousStep != nullptr);
    return *PreviousStep;
  }

  Step &getPredecessor() {
    revng_assert(PreviousStep != nullptr);
    return *PreviousStep;
  }

public:
  llvm::Error runAnalysis(llvm::StringRef AnalysisName,
                          const ContainerToTargetsMap &Targets,
                          const llvm::StringMap<std::string> &ExtraArgs = {});

  /// Executes all the pipes of this step, merges the results in the final
  /// containers and returns the containers filtered according to the request.
  ContainerSet run(ContainerSet &&Targets,
                   const std::vector<PipeExecutionEntry> &ExecutionInfos);

  void pipeInvalidate(const GlobalTupleTreeDiff &Diff,
                      ContainerToTargetsMap &Map) const;

  /// Given the input required goals, calculates backwards how such goals are
  /// achieved by the current step and returns the targets that must be loaded
  /// from the containers in the step before this one, as well as a list of for
  /// each pipe.
  std::pair<ContainerToTargetsMap, std::vector<PipeExecutionEntry>>
  analyzeGoals(const ContainerToTargetsMap &RequiredGoals) const;

  llvm::Error checkPrecondition() const;

  /// Returns the predicted state of the Input containers status after the
  /// execution of all the pipes in this step.
  ContainerToTargetsMap deduceResults(ContainerToTargetsMap Input) const;

public:
  void addPipe(PipeWrapper Wrapper) { Pipes.push_back(std::move(Wrapper)); }

  /// Drops from the backing containers all the targets presents in containers
  /// status
  llvm::Error invalidate(const ContainerToTargetsMap &ToRemove);

public:
  llvm::Error store(const revng::DirectoryPath &DirPath) const;
  llvm::Error load(const revng::DirectoryPath &DirPath);

  std::vector<revng::FilePath>
  getWrittenFiles(const revng::DirectoryPath &DirPath) const;

public:
  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const {
    indent(OS, Indentation);
    OS << "Step " << Name << ":\n";

    indent(OS, Indentation + 1);
    OS << "Pipes: \n";
    for (const PipeWrapper &Pipe : Pipes)
      Pipe.Pipe.dump(OS, Indentation + 2);

    indent(OS, Indentation + 1);
    OS << " containers: \n";
    Containers.dump(OS, Indentation + 2);
  }

  void dump() const debug_function { dump(dbg); }

private:
  static void removeSatisfiedGoals(TargetsList &RequiredInputs,
                                   const ContainerBase &CachedSymbols,
                                   TargetsList &ToLoad);

private:
  void removeSatisfiedGoals(ContainerToTargetsMap &Targets,
                            ContainerToTargetsMap &ToLoad) const;

  void explainExecutedPipe(const InvokableWrapperBase &Wrapper,
                           size_t Indentation = 0) const;
  void explainStartStep(const ContainerToTargetsMap &Wrapper,
                        size_t Indentation = 0) const;
  void explainEndStep(const ContainerToTargetsMap &Wrapper,
                      size_t Indentation = 0) const;
};

} // namespace pipeline
