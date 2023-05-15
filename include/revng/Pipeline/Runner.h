#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/AnalysesList.h"
#include "revng/Pipeline/ContainerFactorySet.h"
#include "revng/Pipeline/GlobalTupleTreeDiff.h"
#include "revng/Pipeline/KindsRegistry.h"
#include "revng/Pipeline/Step.h"
#include "revng/Pipeline/Target.h"
#include "revng/Support/Debug.h"

namespace pipeline {

class Context;
/// A Runner is a wrapper around a pipeline structure and the context needed to
/// run it.
/// It is the top level object on which to invoke operations.
class Runner {
private:
  using Map = llvm::StringMap<Step>;
  using Vector = std::vector<Step *>;

private:
  Context *TheContext;
  ContainerFactorySet ContainerFactoriesRegistry;
  bool IsContainerFactoriesRegistryFinalized = false;

  Map Steps;
  Vector ReversePostOrderIndexes;
  llvm::StringMap<AnalysesList> AnalysesLists;

public:
  template<typename T>
  using DereferenceIteratorType = ::revng::DereferenceIteratorType<T>;
  using iterator = DereferenceIteratorType<Vector::iterator>;
  using const_iterator = DereferenceIteratorType<Vector::const_iterator>;

  using State = llvm::StringMap<ContainerToTargetsMap>;

public:
  explicit Runner(Context &C) : TheContext(&C) {}

public:
  void getCurrentState(State &Out) const;

public:
  size_t registeredContainersCount() const {
    return ContainerFactoriesRegistry.size();
  }

  const ContainerFactorySet &getContainerFactorySet() const {
    return ContainerFactoriesRegistry;
  }

  const Context &getContext() const { return *TheContext; }

  const KindsRegistry &getKindsRegistry() const;

  llvm::Error apply(const GlobalTupleTreeDiff &Diff,
                    pipeline::InvalidationMap &Map);
  void getDiffInvalidations(const GlobalTupleTreeDiff &Diff,
                            pipeline::InvalidationMap &Out) const;

public:
  Step &operator[](llvm::StringRef Name) { return getStep(Name); }
  const Step &operator[](llvm::StringRef Name) const { return getStep(Name); }

  Step &getStep(llvm::StringRef Name) {
    revng_assert(Steps.find(Name) != Steps.end());
    return Steps.find(Name)->second;
  }

  const Step &getStep(llvm::StringRef Name) const {
    revng_assert(Steps.find(Name) != Steps.end());
    return Steps.find(Name)->second;
  }

  bool containsAnalysis(llvm::StringRef Name) const {
    for (const auto &Step : Steps)
      for (const auto &Analysis : Step.second.analyses())
        if (Analysis.first() == Name)
          return true;

    return false;
  }

public:
  bool hasAnalysesList(llvm::StringRef Name) const {
    return AnalysesLists.count(Name);
  }

  size_t getAnalysesListCount() const { return AnalysesLists.size(); }
  const AnalysesList &getAnalysesList(size_t Index) const {
    return std::next(AnalysesLists.begin(), Index)->second;
  }

  const AnalysesList &getAnalysesList(llvm::StringRef Name) const {
    return AnalysesLists.find(Name)->second;
  }

  void addAnalysesList(llvm::StringRef Name,
                       llvm::ArrayRef<AnalysisReference> Analyses) {
    revng_assert(not hasAnalysesList(Name));
    AnalysesLists.try_emplace(Name, pipeline::AnalysesList(Name, Analyses));
  }

public:
  /// Given a target, all occurrences of that target from every container in
  /// every step will be registered in the returned invalidation map. The
  /// propagations will not be calculated.
  llvm::Error getInvalidations(const Target &Target,
                               pipeline::InvalidationMap &Invalidations) const;

  /// Deduces and register in the invalidation map all the targets that have
  /// been produced starting from targets already presents in the map.
  llvm::Error getInvalidations(pipeline::InvalidationMap &Invalidated) const;

public:
  template<typename... PipeWrappers>
  Step &emplaceStep(llvm::StringRef PreviousStepName,
                    llvm::StringRef StepName,
                    PipeWrappers &&...Wrappers) {
    IsContainerFactoriesRegistryFinalized = true;
    if (PreviousStepName.empty())
      return addStep(Step(StepName.str(),
                          ContainerFactoriesRegistry.createEmpty(),
                          std::forward<PipeWrappers>(Wrappers)...));
    else
      return addStep(Step(StepName.str(),
                          ContainerFactoriesRegistry.createEmpty(),
                          operator[](PreviousStepName),
                          std::forward<PipeWrappers>(Wrappers)...));
  }

  Step &addStep(Step &&NewStep);

  llvm::Error run(llvm::StringRef EndingStepName,
                  const ContainerToTargetsMap &Targets);

  llvm::Error run(const State &ToProduce) {
    for (const auto &Request : ToProduce)
      if (auto Error = run(Request.first(), Request.second))
        return Error;

    return llvm::Error::success();
  }

  AnalysisWrapper *findAnalysis(llvm::StringRef AnalysisName) {
    for (auto &Step : Steps) {
      if (Step.second.hasAnalysis(AnalysisName))
        return &Step.second.getAnalysis(AnalysisName);
    }

    return nullptr;
  }

  llvm::Expected<DiffMap>
  runAnalysis(llvm::StringRef AnalysisName,
              llvm::StringRef StepName,
              const ContainerToTargetsMap &Targets,
              pipeline::InvalidationMap &InvalidationsMap,
              const llvm::StringMap<std::string> &Options = {});

  llvm::Expected<DiffMap>
  runAnalyses(const AnalysesList &List,
              pipeline::InvalidationMap &InvalidationsMap,
              const llvm::StringMap<std::string> &Options = {});

  void addContainerFactory(llvm::StringRef Name, ContainerFactory Entry) {
    ContainerFactoriesRegistry.registerContainerFactory(Name, std::move(Entry));
  }

  /// Prefer this overload when the container you wish to add is default
  /// constructible. This should be most of the time, but sometimes this is not
  /// possible, in particular when some context information is needed such as
  /// llvm containers that require the llvm context to be constructed.
  template<typename Container>
  void addDefaultConstructibleFactory(llvm::StringRef Name) {
    auto *Message = "you can only registers containers before adding a step, "
                    "otherwise the already present steps will not be aware of "
                    "the newly registered containers.";
    revng_assert(not IsContainerFactoriesRegistryFinalized, Message);
    auto &Registry = ContainerFactoriesRegistry;
    Registry.registerDefaultConstructibleFactory<Container>(Name);
  }

public:
  /// Remove the provided target from all containers in all the steps, as well
  /// as all all their transitive dependencies
  llvm::Error invalidate(const Target &Target);
  llvm::Error invalidate(const pipeline::InvalidationMap &Invalidations);

public:
  llvm::Error storeToDisk(llvm::StringRef DirPath) const;
  llvm::Error storeToDiskDebug(const char *DirPath) const debug_function {
    return storeToDisk(DirPath);
  }
  llvm::Error storeStepToDisk(llvm::StringRef StepName,
                              llvm::StringRef DirPath) const;
  llvm::Error loadFromDisk(llvm::StringRef DirPath);

public:
  void deduceAllPossibleTargets(State &State) const;

public:
  bool hasSuccessors(llvm::StringRef Name) const {
    return hasSuccessors(operator[](Name));
  }

  bool hasSuccessors(const Step &Current) const {
    const auto IsSuccessor = [&Current](const Step &MaybeSucessor) {
      if (not MaybeSucessor.hasPredecessor())
        return false;

      return &Current == &MaybeSucessor.getPredecessor();
    };

    return llvm::any_of(*this, IsSuccessor);
  }

  bool containsStep(llvm::StringRef Name) const {
    return llvm::any_of(*this, [&Name](const Step &Step) {
      return Step.getName() == Name;
    });
  }

public:
  iterator begin() {
    return ::revng::dereferenceIterator(ReversePostOrderIndexes.begin());
  }
  iterator end() {
    return ::revng::dereferenceIterator(ReversePostOrderIndexes.end());
  }

  const_iterator begin() const {
    return ::revng::dereferenceIterator(ReversePostOrderIndexes.begin());
  }
  const_iterator end() const {
    return ::revng::dereferenceIterator(ReversePostOrderIndexes.end());
  }

  size_t size() const { return Steps.size(); }

public:
  template<typename OStream>
  void dump(OStream &OS, size_t Indentation = 0) const {
    for (const auto &Step : Steps) {
      indent(OS, Indentation);
      OS << Step.first().str() << "\n";
      Step.second.dump(OS, Indentation);
    }
  }

  void dump() const debug_function { dump(dbg); }
};

class PipelineFileMapping {
private:
  std::string Step;
  std::string Container;
  std::string InputFile;

public:
  PipelineFileMapping(llvm::StringRef Step,
                      llvm::StringRef Container,
                      llvm::StringRef InputFile) :
    Step(Step.str()), Container(Container.str()), InputFile(InputFile.str()) {}

public:
  static llvm::Expected<PipelineFileMapping> parse(llvm::StringRef ToParse);

public:
  llvm::Error loadFromDisk(Runner &LoadInto) const;
  llvm::Error storeToDisk(Runner &LoadInto) const;
};

} // namespace pipeline
