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

public:
  template<typename... PipeWrapperTypes>
  Step(std::string Name,
       std::string Component,
       ContainerSet Containers,
       PipeWrapperTypes &&...PipeWrappers) :
    Name(std::move(Name)),
    Component(std::move(Component)),
    Containers(std::move(Containers)),
    Pipes({ std::forward<PipeWrapperTypes>(PipeWrappers)... }),
    PreviousStep(nullptr) {}

  template<typename... PipeWrapperTypes>
  Step(std::string Name,
       std::string Component,
       ContainerSet Containers,
       Step &PreviousStep,
       PipeWrapperTypes &&...PipeWrappers) :
    Name(std::move(Name)),
    Component(std::move(Component)),
    Containers(std::move(Containers)),
    Pipes({ std::forward<PipeWrapperTypes>(PipeWrappers)... }),
    PreviousStep(&PreviousStep) {}

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

    auto &ContainerName = Artifacts.Container;
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
                          Context &Ctx,
                          const ContainerToTargetsMap &Targets,
                          const llvm::StringMap<std::string> &ExtraArgs = {});

  /// Executes all the pipes of this step, merges the results in the final
  /// containers and returns the containers filtered according to the request.
  ContainerSet run(Context &Ctx, ContainerSet &&Targets);

  /// Returns the set of goals that are already contained in the backing
  /// containers of this step, furthermore adds to the container ToLoad those
  /// that were not present.
  ContainerToTargetsMap
  analyzeGoals(const Context &Ctx,
               const ContainerToTargetsMap &RequiredGoals) const;

  llvm::Error checkPrecondition(const Context &Ctx) const;

  /// Returns the predicted state of the Input containers status after the
  /// execution of all the pipes in this step.
  ContainerToTargetsMap deduceResults(const Context &Ctx,
                                      ContainerToTargetsMap Input) const;

public:
  void addPipe(PipeWrapper Wrapper) { Pipes.push_back(std::move(Wrapper)); }

  /// Drops from the backing containers all the targets presents in containers
  /// status
  llvm::Error invalidate(const ContainerToTargetsMap &ToRemove);

public:
  llvm::Error store(const revng::DirectoryPath &DirPath) const;
  llvm::Error load(const revng::DirectoryPath &DirPath);

public:
  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const {
    indent(OS, Indentation);
    OS << "Step " << Name << ":\n";

    indent(OS, Indentation + 1);
    OS << "Pipes: \n";
    for (const auto &Pipe : Pipes)
      Pipe.dump(OS, Indentation + 2);

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

  void explainExecutedPipe(const Context &Ctx,
                           const InvokableWrapperBase &Wrapper,
                           size_t Indentation = 0) const;
  void explainStartStep(const ContainerToTargetsMap &Wrapper,
                        size_t Indentation = 0) const;
  void explainEndStep(const ContainerToTargetsMap &Wrapper,
                      size_t Indentation = 0) const;
};

} // namespace pipeline
