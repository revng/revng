#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipes/ModelGlobal.h"

namespace revng::pipes {

/// This is a "God Object" that can be used when there will be exactly one
/// Pipeline spawned by this process.
///
/// It will care of all configurations including llvm ones and provide a runner
/// in a usable state.
class PipelineManager {
private:
  explicit PipelineManager() = default;

  std::string ExecutionDirectory;
  /// The various member here MUST be unique_ptr to ensure that the various
  /// pointer that go from one to the other are stable, Since there a create
  /// method that returns a expected<PipelineManager>, this is the only way to
  /// ensure this is correct.
  std::unique_ptr<llvm::LLVMContext> Context;
  std::unique_ptr<pipeline::Context> PipelineContext;
  std::unique_ptr<pipeline::Loader> Loader;
  std::unique_ptr<pipeline::Runner> Runner;
  pipeline::Runner::State CurrentState;
  std::map<const pipeline::ContainerSet::value_type *,
           const pipeline::TargetsList *>
    ContainerToEnumeration;

  static llvm::Expected<PipelineManager>
  createContexts(llvm::ArrayRef<std::string> EnablingFlags,
                 llvm::StringRef ExecutionDirectory);

public:
  PipelineManager(PipelineManager &&Other) = default;
  PipelineManager &operator=(PipelineManager &&Other) = default;
  PipelineManager &operator=(const PipelineManager &Other) = delete;
  PipelineManager(const PipelineManager &Other) = delete;
  ~PipelineManager() = default;

  const pipeline::Kind *getKind(llvm::StringRef Name) const {
    return Runner->getKindsRegistry().find(Name);
  }

  /// Tries to set up a PipelineManager with the provided files pipelines,
  /// enabling flags loaded from the ExecutionDirectory. If anything is invalid
  /// a Error is returned instead.
  static llvm::Expected<PipelineManager>
  create(llvm::ArrayRef<std::string> PipelinePath,
         llvm::ArrayRef<std::string> EnablingFlags,
         llvm::StringRef ExecutionDirectory);

  /// Exactly like create except the pipeline are not provided as paths but
  /// directly as yaml file buffer in memory.
  static llvm::Expected<PipelineManager>
  createFromMemory(llvm::ArrayRef<std::string> InMemoryPipeline,
                   llvm::ArrayRef<std::string> EnablingFlags,
                   llvm::StringRef ExecutionDirectory);

  /// Entirelly replaces the container indicated by the mapping with the file
  /// indicated by the mapping
  llvm::Error overrideContainer(pipeline::PipelineFileMapping Mapping);
  /// The same as the previous overload except the mapping is provided as a
  /// triple to be parsed is the usual way.
  llvm::Error overrideContainer(llvm::StringRef PipelineFileMapping);

  /// Stores the content of the container indicated by the mapping at the path
  /// indicated by the mapping, and  nothing else.
  llvm::Error store(const pipeline::PipelineFileMapping &StoresOverride);

  /// Builds FileMapping out of each provided string and the invoke store on
  /// each of them.
  llvm::Error store(llvm::ArrayRef<std::string> StoresOverrides);

  /// Triggers the full serialization of every step and every container to the
  /// the specified DirPath or the Execution directory if omitted.
  llvm::Error storeToDisk(llvm::StringRef DirPath = llvm::StringRef());

  /// Trigger the serialization of a single Step to the specified DirPath or
  /// the Execution directory if omitted.
  llvm::Error storeStepToDisk(llvm::StringRef StepName,
                              llvm::StringRef DirPath = llvm::StringRef());

  const pipeline::Step::AnalysisValueType &
  getAnalysis(const pipeline::AnalysisReference &Reference) const;

  llvm::Error deserializeContainer(pipeline::Step &Step,
                                   llvm::StringRef ContainerName,
                                   const llvm::MemoryBuffer &Buffer);

  const pipeline::Context &context() const { return *PipelineContext; }

  pipeline::Context &context() { return *PipelineContext; }

  /// recalculates the current available targetsd and keeps overship of the
  /// computer info
  void recalculateCurrentState();

  /// recalculates the cache for fast lookups
  void recalculateCache();

  /// like recalculate by the ownerhip is maintained by State
  void getAllPossibleTargets(pipeline::Runner::State &State,
                             bool ExpandTargets = true) const;
  /// like recalculate by the ownerhip is maintained by State
  void getCurrentState(pipeline::Runner::State &State) const;

  /// returns a reference to the internal Runner::State populated by recalculate
  /// memethods.
  const pipeline::Runner::State &getLastState() const { return CurrentState; }

  /// A helper function used to produce all possible targets. It is used for
  /// debug purposes to see if any particular target crashes.
  llvm::Error produceAllPossibleTargets() {
    return produceAllPossibleTargets(false);
  }
  llvm::Error produceAllPossibleSingleTargets() {
    return produceAllPossibleTargets(true);
  }

  llvm::Expected<pipeline::InvalidationMap> invalidateAllPossibleTargets();
  llvm::Expected<pipeline::InvalidationMap>
  invalidateFromDiff(const llvm::StringRef Name,
                     const pipeline::GlobalTupleTreeDiff &Diff);

  /// returns the cached list of targets that are known to be available to be
  /// produced in a container
  const pipeline::TargetsList *
  getTargetsAvailableFor(const pipeline::ContainerSet::value_type &Container) {
    if (auto Iter = ContainerToEnumeration.find(&Container);
        Iter == ContainerToEnumeration.end())
      return nullptr;
    else
      return Iter->second;
  }

  void dump() const { Runner->dump(); }

  const pipeline::Runner &getRunner() const { return *Runner; }
  pipeline::Runner &getRunner() { return *Runner; }

  llvm::Expected<pipeline::DiffMap>
  runAnalyses(const pipeline::AnalysesList &List,
              pipeline::InvalidationMap &Map,
              const llvm::StringMap<std::string> &Options = {},
              llvm::raw_ostream *DiagnosticLog = nullptr);

  llvm::Expected<pipeline::DiffMap>
  runAnalysis(llvm::StringRef AnalysisName,
              llvm::StringRef StepName,
              const pipeline::ContainerToTargetsMap &Targets,
              pipeline::InvalidationMap &Map,
              const llvm::StringMap<std::string> &Options = {},
              llvm::raw_ostream *DiagnosticLog = nullptr);

  /// Run all analysis in reverse post order (that is: parents first),
  llvm::Expected<pipeline::DiffMap>
  runAllAnalyses(pipeline::InvalidationMap &Map,
                 const llvm::StringMap<std::string> &Options = {});

  /// recalculates all possible targets and keeps overship of the computed info
  void recalculateAllPossibleTargets(bool ExpandTargets = true);

  /// prints to the provided raw_ostream all possible targets that can
  /// be produced by the pipeline in the current state
  void writeAllPossibleTargets(llvm::raw_ostream &OS) const;

  llvm::StringRef executionDirectory() const { return ExecutionDirectory; }

private:
  llvm::Error produceAllPossibleTargets(bool ExpandTargets);
};
} // namespace revng::pipes
