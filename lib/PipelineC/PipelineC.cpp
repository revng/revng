//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"
#include "revng/PipelineC/PipelineC.h"
#include "revng/PipelineC/Tracing/Private.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/PipelineManager.h"
#include "revng/Support/Assert.h"
#include "revng/Support/InitRevng.h"
#include "revng/TupleTree/TupleTreeDiff.h"

#include "Tracing/Wrapper.h"

using namespace pipeline;
using namespace ::revng::pipes;
namespace cl = llvm::cl;

template<typename T>
concept default_constructible = std::is_default_constructible<T>::value;

template<default_constructible T>
class ExistingOrNew {
private:
  T *Pointer;
  std::optional<T> Default;

public:
  explicit ExistingOrNew(T *Pointer) {
    if (Pointer != nullptr) {
      this->Pointer = Pointer;
    } else {
      Default.emplace();
      this->Pointer = &*Default;
    }
  }

  ExistingOrNew(ExistingOrNew &&) = delete;
  ExistingOrNew(const ExistingOrNew &) = delete;
  ExistingOrNew &operator=(ExistingOrNew &&) = delete;
  ExistingOrNew &operator=(const ExistingOrNew &) = delete;
  ~ExistingOrNew() = default;

  T &operator*() { return *Pointer; }
  T *operator->() { return Pointer; }
};

void revng::tracing::setTracing(llvm::raw_ostream *OS) {
  Tracing.swap(OS);
}

/// Used when we want to return a stack allocated string. Copies the string onto
/// the heap and gives ownership of to the caller
static char *copyString(llvm::StringRef Str) {
  char *ToReturn = (char *) malloc(sizeof(char) * (Str.size() + 1));
  strncpy(ToReturn, Str.data(), Str.size());
  ToReturn[Str.size()] = 0;
  return ToReturn;
}

static bool Initialized = false;
static std::optional<revng::InitRevng> InitRevngInstance = std::nullopt;
static cl::list<std::string> PipelinePaths("pipeline-path", cl::ZeroOrMore);
typedef void (*sighandler_t)(int);

// NOLINTBEGIN

static bool _rp_initialize(int argc,
                           const char *argv[],
                           uint32_t signals_to_preserve_count,
                           int signals_to_preserve[]) {
  if (argc != 0)
    revng_check(argv != nullptr);

  if (Initialized)
    return false;

  std::map<int, sighandler_t> Signals;
  for (uint32_t I = 0; I < signals_to_preserve_count; I++) {
    // For each signal number we are asked to preserve we need to extract the
    // function pointer to the signal and save it. The constructor for
    // revng::InitRevng will call LLVM's RegisterHandlers which overrides most
    // signal handlers and chains the previous one afterwards, we want instead
    // to keep the already existing handler and remove the LLVM's one
    int SigNumber = signals_to_preserve[I];
    sighandler_t Handler = signal(SigNumber, SIG_DFL);
    if (Handler != SIG_ERR && Handler != NULL) {
      // We save the signal handler for restoration after we initialize LLVM's
      // machinery, as said we do not restore the signal to avoid LLVM chaining
      // it after its own
      Signals[SigNumber] = Handler;
    }
  }

  revng_check(not InitRevngInstance.has_value());
  InitRevngInstance.emplace(argc,
                            argv,
                            "",
                            llvm::ArrayRef<llvm::cl::OptionCategory *>());

  for (const auto &[SigNumber, Handler] : Signals) {
    // All of LLVM's initialization is complete, restore the original signals to
    // the respective signal number
    signal(SigNumber, Handler);
  }

  Initialized = true;
  Registry::runAllInitializationRoutines();

  return true;
}

static bool _rp_shutdown() {
  if (InitRevngInstance.has_value()) {
    InitRevngInstance.reset();
    return true;
  }
  return false;
}

static rp_manager *rp_manager_create_impl(llvm::ArrayRef<std::string> Pipelines,
                                          uint64_t pipeline_flags_count,
                                          const char *pipeline_flags[],
                                          const char *execution_directory,
                                          bool is_path = true) {
  revng_check(Initialized);
  if (pipeline_flags == nullptr)
    revng_check(pipeline_flags_count == 0);
  revng_check(execution_directory != nullptr);

  std::vector<std::string> FlagsVector;
  for (size_t I = 0; I < pipeline_flags_count; I++)
    FlagsVector.push_back(pipeline_flags[I]);

  auto Pipeline = is_path ?
                    PipelineManager::create(Pipelines,
                                            FlagsVector,
                                            execution_directory) :
                    PipelineManager::createFromMemory(Pipelines,
                                                      FlagsVector,
                                                      execution_directory);
  if (not Pipeline) {
    auto Error = Pipeline.takeError();
    llvm::errs() << Error;
    llvm::consumeError(std::move(Error));
    return nullptr;
  }

  auto manager = new PipelineManager(std::move(*Pipeline));
  return manager;
}

static rp_manager *
_rp_manager_create_from_string(uint64_t pipelines_count,
                               const char *pipelines[],
                               uint64_t pipeline_flags_count,
                               const char *pipeline_flags[],
                               const char *execution_directory) {
  std::vector<std::string> Pipelines;
  for (size_t I = 0; I < pipelines_count; I++)
    Pipelines.push_back(pipelines[I]);

  return rp_manager_create_impl(Pipelines,
                                pipeline_flags_count,
                                pipeline_flags,
                                execution_directory,
                                false);
}

static rp_manager *_rp_manager_create(uint64_t pipeline_flags_count,
                                      const char *pipeline_flags[],
                                      const char *execution_directory) {
  return rp_manager_create_impl(PipelinePaths,
                                pipeline_flags_count,
                                pipeline_flags,
                                execution_directory,
                                true);
}

static bool _rp_manager_save(rp_manager *manager) {
  revng_check(manager != nullptr);

  auto Error = manager->storeToDisk();
  if (not Error)
    return true;

  llvm::consumeError(std::move(Error));
  return false;
}

static void _rp_manager_destroy(rp_manager *manager) {
  revng_check(manager != nullptr);
  delete manager;
}

static uint64_t _rp_manager_steps_count(const rp_manager *manager) {
  revng_check(manager != nullptr);
  return manager->getRunner().size();
}

static uint64_t _rp_manager_step_name_to_index(const rp_manager *manager,
                                               const char *name) {
  revng_check(manager != nullptr);
  revng_check(name != nullptr);
  size_t I = 0;
  for (const auto &Step : manager->getRunner()) {
    if (Step.getName() == name)
      return I;
    I++;
  }

  return RP_STEP_NOT_FOUND;
}

static rp_step *_rp_manager_get_step(const rp_manager *manager,
                                     uint64_t index) {
  revng_check(manager != nullptr);
  revng_check(index < manager->getRunner().size());
  return &(*(std::next(manager->getRunner().begin(), index)));
}

static const char *_rp_step_get_name(const rp_step *step) {
  revng_check(step != nullptr);
  return step->getName().data();
}

static uint64_t _rp_step_get_analyses_count(const rp_step *step) {
  return step->getAnalysesSize();
}

static const rp_analysis *_rp_step_get_analysis(const rp_step *step,
                                                uint64_t index) {
  revng_check(step != nullptr);
  if (index >= step->getAnalysesSize())
    return nullptr;

  return &*(std::next(step->analysesBegin(), index));
}

static rp_container *
_rp_step_get_container(rp_step *step, rp_container_identifier *container) {
  revng_check(step != nullptr);
  revng_check(container != nullptr);
  if (step->containers().isContainerRegistered(container->first())) {
    step->containers()[container->first()];
    return &*step->containers().find(container->first());
  } else {
    return nullptr;
  }
}

static const rp_kind *_rp_step_get_artifacts_kind(const rp_step *step) {
  revng_check(step != nullptr);
  return step->getArtifactsKind();
}

static const rp_container *_rp_step_get_artifacts_container(rp_step *step) {
  revng_check(step != nullptr);
  return step->getArtifactsContainer();
}

static char *
_rp_step_get_artifacts_single_target_filename(const rp_step *step) {
  revng_check(step != nullptr);
  return copyString(step->getArtifactsSingleTargetFilename());
}

static uint64_t
_rp_targets_list_targets_count(const rp_targets_list *targets_list) {
  revng_check(targets_list != nullptr);
  return targets_list->size();
}

static rp_kind *_rp_manager_get_kind_from_name(const rp_manager *manager,
                                               const char *kind_name) {
  revng_check(manager != nullptr);
  revng_check(kind_name != nullptr);

  return manager->getKind(kind_name);
}

static const char *_rp_kind_get_name(const rp_kind *kind) {
  revng_check(kind != nullptr);
  return kind->name().data();
}

static const rp_kind *_rp_kind_get_parent(const rp_kind *kind) {
  revng_check(kind != nullptr);
  return kind->parent();
}

static rp_diff_map *
_rp_manager_run_analysis(rp_manager *manager,
                         const char *step_name,
                         const char *analysis_name,
                         const rp_container_targets_map *target_map,
                         rp_invalidations *invalidations,
                         const rp_string_map *options) {
  revng_check(manager != nullptr);
  revng_check(step_name != nullptr);
  revng_check(analysis_name != nullptr);
  revng_check(target_map != nullptr);

  ExistingOrNew<rp_invalidations> Invalidations(invalidations);
  ExistingOrNew<const rp_string_map> Options(options);

  auto MaybeDiffs = manager->runAnalysis(analysis_name,
                                         step_name,
                                         *target_map,
                                         *Invalidations,
                                         *Options);
  if (!MaybeDiffs) {
    llvm::consumeError(MaybeDiffs.takeError());
    return nullptr;
  }

  return new rp_diff_map(std::move(*MaybeDiffs));
}

static void _rp_diff_map_destroy(rp_diff_map *to_free) {
  delete to_free;
}

static char *_rp_diff_map_get_diff(const rp_diff_map *map,
                                   const char *global_name) {
  auto It = map->find(global_name);
  if (It == map->end())
    return nullptr;

  std::string S;
  llvm::raw_string_ostream OS(S);
  It->second.serialize(OS);
  OS.flush();
  return copyString(S);
}

static rp_buffer *_rp_manager_produce_targets(rp_manager *manager,
                                              uint64_t targets_count,
                                              rp_target *targets[],
                                              const rp_step *step,
                                              const rp_container *container) {
  revng_check(manager != nullptr);
  revng_check(targets_count != 0);
  revng_check(targets != nullptr);
  revng_check(step != nullptr);
  revng_check(container != nullptr);

  TargetsList List;
  for (size_t I = 0; I < targets_count; I++)
    List.push_back(*targets[I]);

  auto ErrorOrCloned = manager->produceTargets(step->getName(),
                                               *container,
                                               List);

  if (!ErrorOrCloned) {
    llvm::consumeError(ErrorOrCloned.takeError());
    return nullptr;
  }

  auto &Cloned = ErrorOrCloned.get();
  rp_buffer *Out = new rp_buffer();
  llvm::raw_svector_ostream Serialized(*Out);
  llvm::cantFail(Cloned->serialize(Serialized));

  return Out;
}

static rp_target *_rp_target_create(const rp_kind *kind,
                                    uint64_t path_components_count,
                                    const char *path_components[]) {
  revng_check(kind != nullptr);
  revng_check(path_components != nullptr);
  revng_check(kind->rank().depth() == path_components_count);
  std::vector<std::string> List;
  for (size_t I = 0; I < path_components_count; I++) {
    llvm::StringRef Rank(path_components[I]);
    List.push_back(Rank.str());
  }

  return new Target(std::move(List), *kind);
}

static void _rp_target_destroy(rp_target *target) {
  revng_check(target != nullptr);
  delete target;
}

static uint64_t _rp_manager_kinds_count(const rp_manager *manager) {
  revng_check(manager != nullptr);
  return manager->getRunner().getKindsRegistry().size();
}

static rp_kind *_rp_manager_get_kind(const rp_manager *manager,
                                     uint64_t index) {
  revng_check(manager != nullptr);
  revng_check(index < manager->getRunner().getKindsRegistry().size());
  return &*std::next(manager->getRunner().getKindsRegistry().begin(), index);
}

static bool _rp_container_store(const rp_container *container,
                                const char *path) {
  revng_check(container != nullptr);
  revng_check(path != nullptr);
  auto Error = container->second->storeToDisk(path);
  if (not Error)
    return true;

  llvm::consumeError(std::move(Error));
  return false;
}

static bool _rp_manager_container_deserialize(rp_manager *manager,
                                              rp_step *step,
                                              const char *container_name,
                                              const char *content,
                                              uint64_t size) {
  revng_check(manager != nullptr);
  revng_check(step != nullptr);
  revng_check(container_name != nullptr);
  revng_check(content != nullptr);

  auto Buffer = llvm::MemoryBuffer::getMemBuffer(llvm::StringRef(content, size),
                                                 "",
                                                 false);
  auto Error = manager->deserializeContainer(*step, container_name, *Buffer);
  if (!!Error) {
    llvm::consumeError(std::move(Error));
    return false;
  }
  return true;
}

static rp_kind *_rp_target_get_kind(rp_target *target) {
  revng_check(target != nullptr);
  return &target->getKind();
}

static uint64_t _rp_target_path_components_count(rp_target *target) {
  revng_check(target != nullptr);
  return target->getPathComponents().size();
}
static const char *_rp_target_get_path_component(rp_target *target,
                                                 uint64_t index) {
  revng_check(target != nullptr);
  auto &PathComponents = target->getPathComponents();
  revng_check(index < PathComponents.size());

  return PathComponents[index].c_str();
}

static char *_rp_manager_create_container_path(rp_manager *manager,
                                               const char *step_name,
                                               const char *container_name) {
  revng_check(manager != nullptr);
  revng_check(step_name != nullptr);
  revng_check(container_name != nullptr);

  if (manager->executionDirectory().empty())
    return nullptr;

  llvm::SmallString<128> Path;
  llvm::sys::path::append(Path,
                          manager->executionDirectory(),
                          step_name,
                          container_name);
  return copyString(Path);
}

static rp_targets_list *
_rp_manager_get_container_targets_list(const rp_manager *manager,
                                       const rp_container *container) {
  revng_check(manager != nullptr);
  revng_check(container != nullptr);

  return manager->getTargetsAvailableFor(*container);
}

static rp_target *
_rp_targets_list_get_target(const rp_targets_list *targets_list,
                            uint64_t index) {
  revng_check(targets_list != nullptr);
  revng_check(index < targets_list->size());
  return new rp_target((*targets_list)[index]);
}

static void _rp_string_destroy(char *string) {
  revng_check(string != nullptr);
  free(string);
}

static uint64_t _rp_manager_containers_count(const rp_manager *manager) {
  revng_check(manager != nullptr);
  return manager->getRunner().registeredContainersCount();
}

static rp_container_identifier *
_rp_manager_get_container_identifier(const rp_manager *manager,
                                     uint64_t index) {
  revng_check(manager != nullptr);
  const auto &ContainerRegistry = manager->getRunner().getContainerFactorySet();
  revng_check(index < ContainerRegistry.size());
  return &*std::next(ContainerRegistry.begin(), index);
}

static const char *_rp_container_get_name(const rp_container *container) {
  revng_check(container != nullptr);
  return container->second->name().data();
}

static const char *
_rp_container_identifier_get_name(const rp_container_identifier *identifier) {
  revng_check(identifier != nullptr);
  return identifier->first().data();
}

static char *_rp_target_create_serialized_string(rp_target *target) {
  revng_check(target != nullptr);
  return copyString(target->serialize());
}

static rp_target *_rp_target_create_from_string(const rp_manager *manager,
                                                const char *string) {
  revng_check(manager != nullptr);
  revng_check(string != nullptr);
  TargetsList Targets;
  auto Error = parseTarget(manager->context(),
                           string,
                           manager->context().getKindsRegistry(),
                           Targets);
  revng_check(Targets.size() == 1);
  if (not Error)
    return new Target(std::move(Targets.front()));

  llvm::consumeError(std::move(Error));
  return nullptr;
}

static bool _rp_target_is_ready(const rp_target *target,
                                const rp_container *container) {
  revng_assert(target);
  revng_assert(container);
  return container->second->enumerate().contains(*target);
}

/// TODO Remove the redundant copy by writing a custom string stream that writes
/// directly to a buffer to return.
static char *_rp_manager_create_global_copy(const rp_manager *manager,
                                            const char *global_name) {
  std::string Out;
  llvm::raw_string_ostream Serialized(Out);
  auto &GlobalsMap = manager->context().getGlobals();
  if (auto Error = GlobalsMap.serialize(global_name, Serialized); Error) {
    llvm::consumeError(std::move(Error));
    return nullptr;
  }
  Serialized.flush();
  return copyString(Out);
}

static uint64_t _rp_manager_get_globals_count(rp_manager *manager) {
  return manager->context().getGlobals().size();
}

static char *_rp_manager_get_global_name(const rp_manager *manager,
                                         uint64_t index) {
  if (index < manager->context().getGlobals().size())
    return copyString(manager->context().getGlobals().getName(index).str());
  return nullptr;
}

static bool llvmErrorToRpError(llvm::Error Error,
                               ExistingOrNew<rp_error> &Out) {
  bool Success = true;

  auto DocumentedErrorHandler = [&](const revng::DocumentErrorBase &Error) {
    rp_document_error ErrorBody(Error.getTypeName(),
                                Error.getLocationTypeName());

    for (size_t I = 0; I < Error.size(); I++) {
      rp_error_reason reason(Error.getMessage(I), Error.getLocation(I));
      ErrorBody.Reasons.emplace_back(reason);
    }

    *Out = std::move(ErrorBody);
    Success = false;
  };

  auto OtherErrorHandler = [&](const llvm::ErrorInfoBase &OtherErrors) {
    std::string Reason;
    llvm::raw_string_ostream OS(Reason);
    OtherErrors.log(OS);
    OS.flush();
    *Out = rp_simple_error(Reason, "");
    Success = false;
  };

  llvm::handleAllErrors(std::move(Error),
                        DocumentedErrorHandler,
                        OtherErrorHandler);

  return Success;
}

template<bool commit>
inline bool rp_manager_set_global_impl(rp_manager *manager,
                                       const char *serialized,
                                       const char *global_name,
                                       rp_invalidations *invalidations,
                                       rp_error *error) {
  revng_check(manager != nullptr);
  revng_check(serialized != nullptr);
  revng_check(global_name != nullptr);

  ExistingOrNew<rp_error> Error(error);
  auto &GlobalsMap = manager->context().getGlobals();
  auto Buffer = llvm::MemoryBuffer::getMemBuffer(serialized);

  auto MaybeGlobal = GlobalsMap.get(global_name);
  if (!MaybeGlobal) {
    llvmErrorToRpError(MaybeGlobal.takeError(), Error);
    return false;
  }

  auto MaybeNewGlobal = GlobalsMap.createNew(global_name, *Buffer);
  if (!MaybeNewGlobal) {
    llvmErrorToRpError(MaybeNewGlobal.takeError(), Error);
    return false;
  }

  if (not MaybeNewGlobal->get()->verify()) {
    *Error = rp_simple_error(std::string("Could not verify ") + global_name,
                             "");
    return false;
  }

  if constexpr (commit) {
    auto &NewGlobal = MaybeNewGlobal.get();
    auto Diff = MaybeGlobal.get()->diff(*NewGlobal);
    *MaybeGlobal.get() = *NewGlobal;

    auto MaybeInvalidations = manager->invalidateFromDiff(global_name, Diff);
    if (!MaybeInvalidations) {
      llvmErrorToRpError(MaybeInvalidations.takeError(), Error);
      return false;
    }

    ExistingOrNew<rp_invalidations> Invalidations(invalidations);
    *Invalidations = MaybeInvalidations.get();
  }

  return true;
}

static bool _rp_manager_set_global(rp_manager *manager,
                                   const char *serialized,
                                   const char *global_name,
                                   rp_invalidations *invalidations,
                                   rp_error *error) {
  return rp_manager_set_global_impl<true>(manager,
                                          serialized,
                                          global_name,
                                          invalidations,
                                          error);
}

static bool _rp_manager_verify_global(rp_manager *manager,
                                      const char *serialized,
                                      const char *global_name,
                                      rp_error *error) {
  return rp_manager_set_global_impl<false>(manager,
                                           serialized,
                                           global_name,
                                           nullptr,
                                           error);
}

template<bool commit>
inline bool rp_manager_apply_diff_impl(rp_manager *manager,
                                       const char *diff,
                                       const char *global_name,
                                       rp_invalidations *invalidations,
                                       rp_error *error) {
  revng_check(manager != nullptr);
  revng_check(diff != nullptr);
  revng_check(global_name != nullptr);

  ExistingOrNew<rp_error> Error(error);
  auto &GlobalsMap = manager->context().getGlobals();
  auto Buffer = llvm::MemoryBuffer::getMemBuffer(diff);

  auto GlobalOrError = GlobalsMap.get(global_name);
  if (!GlobalOrError) {
    llvmErrorToRpError(GlobalOrError.takeError(), Error);
    return false;
  }

  auto &Global = GlobalOrError.get();
  auto MaybeDiff = Global->deserializeDiff(*Buffer);
  if (!MaybeDiff) {
    llvmErrorToRpError(MaybeDiff.takeError(), Error);
    return false;
  }

  auto &Diff = MaybeDiff.get();
  auto GlobalClone = Global->clone();
  auto ApplyError = GlobalClone->applyDiff(Diff);
  if (ApplyError) {
    llvmErrorToRpError(std::move(ApplyError), Error);
    return false;
  }

  if (not GlobalClone->verify()) {
    *Error = rp_simple_error(std::string("could not verify ") + global_name,
                             "");
    return false;
  }

  if constexpr (commit) {
    *Global = *GlobalClone;
    auto MaybeInvalidations = manager->invalidateFromDiff(global_name, Diff);
    if (!MaybeInvalidations) {
      llvmErrorToRpError(MaybeInvalidations.takeError(), Error);
      return false;
    }

    ExistingOrNew<rp_invalidations> Invalidations(invalidations);
    *Invalidations = MaybeInvalidations.get();
  }

  return true;
}

static bool _rp_manager_apply_diff(rp_manager *manager,
                                   const char *diff,
                                   const char *global_name,
                                   rp_invalidations *invalidations,
                                   rp_error *error) {
  return rp_manager_apply_diff_impl<true>(manager,
                                          diff,
                                          global_name,
                                          invalidations,
                                          error);
}

static bool _rp_manager_verify_diff(rp_manager *manager,
                                    const char *diff,
                                    const char *global_name,
                                    rp_error *error) {
  return rp_manager_apply_diff_impl<false>(manager,
                                           diff,
                                           global_name,
                                           nullptr,
                                           error);
}

static uint64_t _rp_ranks_count() {
  return Rank::getAll().size();
}

static const rp_rank *_rp_rank_get(uint64_t index) {
  revng_check(index < Rank::getAll().size());
  return Rank::getAll()[index];
}

static const rp_rank *_rp_rank_get_from_name(const char *rank_name) {
  revng_check(rank_name != nullptr);
  for (auto rank : Rank::getAll()) {
    if (rank->name() == rank_name) {
      return rank;
    }
  }
  return nullptr;
}

static const char *_rp_rank_get_name(const rp_rank *rank) {
  revng_check(rank != nullptr);
  return rank->name().data();
}

static uint64_t _rp_rank_get_depth(const rp_rank *rank) {
  revng_check(rank != nullptr);
  return rank->depth();
}

static const rp_rank *_rp_rank_get_parent(const rp_rank *rank) {
  revng_check(rank != nullptr);
  return rank->parent();
}

static const rp_rank *_rp_kind_get_rank(const rp_kind *kind) {
  revng_check(kind != nullptr);
  return &kind->rank();
}

static const rp_step *_rp_step_get_parent(const rp_step *step) {
  revng_check(step != nullptr);
  if (step->hasPredecessor()) {
    return &step->getPredecessor();
  } else {
    return nullptr;
  }
}

static const char *_rp_container_get_mime(const rp_container *container) {
  revng_check(container != nullptr);
  return container->getValue()->mimeType().data();
}

static rp_buffer *_rp_container_extract_one(const rp_container *container,
                                            const rp_target *target) {
  if (!container->second->enumerate().contains(*target)) {
    return nullptr;
  }

  rp_buffer *Out = new rp_buffer();
  llvm::raw_svector_ostream Serialized(*Out);
  llvm::cantFail(container->second->extractOne(Serialized, *target));

  return Out;
}

static const char *_rp_analysis_get_name(const rp_analysis *analysis) {
  revng_check(analysis != nullptr);
  return analysis->second->getUserBoundName().c_str();
}

static uint64_t _rp_analysis_get_arguments_count(const rp_analysis *analysis) {
  revng_check(analysis != nullptr);
  return analysis->second->getRunningContainersNames().size();
}

static char *_rp_analysis_get_argument_name(const rp_analysis *analysis,
                                            uint64_t index) {
  revng_check(index < analysis->second->getRunningContainersNames().size());
  std::string Name = analysis->second->getRunningContainersNames()[index];

  return copyString(Name);
}

static uint64_t
_rp_analysis_get_argument_acceptable_kinds_count(const rp_analysis *analysis,
                                                 uint64_t argument_index) {
  return analysis->second->getAcceptedKinds(argument_index).size();
}

static const rp_kind *
_rp_analysis_get_argument_acceptable_kind(const rp_analysis *analysis,
                                          uint64_t argument_index,
                                          uint64_t kind_index) {
  const auto &Accepted = analysis->second->getAcceptedKinds(argument_index);
  if (static_cast<size_t>(kind_index) >= Accepted.size())
    return nullptr;
  return Accepted[kind_index];
}

static uint64_t _rp_manager_get_analyses_list_count(rp_manager *manager) {
  revng_check(manager != nullptr);
  return manager->getRunner().getAnalysesListCount();
}

static rp_analyses_list *_rp_manager_get_analyses_list(rp_manager *manager,
                                                       uint64_t index) {
  revng_check(manager != nullptr);
  return &manager->getRunner().getAnalysesList(index);
}

static const char *_rp_analyses_list_get_name(rp_analyses_list *list) {
  revng_check(list != nullptr);
  return list->getName().data();
}

static uint64_t _rp_analyses_list_count(rp_analyses_list *list) {
  revng_check(list != nullptr);
  return list->size();
}

static rp_analysis *_rp_manager_get_analysis(rp_manager *manager,
                                             rp_analyses_list *list,
                                             uint64_t index) {
  revng_check(manager != nullptr);
  revng_check(list != nullptr);
  return &manager->getAnalysis(list->at(index));
}

static rp_diff_map *
_rp_manager_run_analyses_list(rp_manager *manager,
                              rp_analyses_list *list,
                              rp_invalidations *invalidations,
                              const rp_string_map *options) {
  revng_check(manager != nullptr);
  revng_check(list != nullptr);

  ExistingOrNew<rp_invalidations> Invalidations(invalidations);
  ExistingOrNew<const rp_string_map> Options(options);

  auto MaybeDiffs = manager->runAnalyses(*list, *Invalidations, *Options);
  if (!MaybeDiffs) {
    llvm::consumeError(MaybeDiffs.takeError());
    return nullptr;
  }

  return new rp_diff_map(std::move(*MaybeDiffs));
}

static bool _rp_diff_map_is_empty(rp_diff_map *map) {
  revng_check(map != nullptr);
  for (auto &Entry : *map) {
    if (!Entry.second.isEmpty()) {
      return false;
    }
  }
  return true;
}

static rp_error *_rp_error_create() {
  return new rp_error();
}

static bool _rp_error_is_success(const rp_error *error) {
  revng_check(error != nullptr);
  return std::holds_alternative<std::monostate>(*error);
}

static bool _rp_error_is_document_error(const rp_error *error) {
  revng_check(error != nullptr);
  return std::holds_alternative<rp_document_error>(*error);
}

static rp_simple_error *_rp_error_get_simple_error(rp_error *error) {
  revng_check(error != nullptr);
  if (not std::holds_alternative<rp_simple_error>(*error))
    return nullptr;

  return &std::get<rp_simple_error>(*error);
}

static rp_document_error *_rp_error_get_document_error(rp_error *error) {
  revng_check(error != nullptr);
  if (not std::holds_alternative<rp_document_error>(*error))
    return nullptr;

  return &std::get<rp_document_error>(*error);
}

static void _rp_error_destroy(rp_error *error) {
  revng_check(error != nullptr);
  delete error;
}

static size_t _rp_document_error_reasons_count(const rp_document_error *error) {
  revng_check(error != nullptr);
  return error->Reasons.size();
}

static const char *
_rp_document_error_get_error_type(const rp_document_error *error) {
  revng_check(error != nullptr);
  return error->ErrorType.c_str();
}

static const char *
_rp_document_error_get_location_type(const rp_document_error *error) {
  revng_check(error != nullptr);
  return error->LocationType.c_str();
}

static const char *
_rp_document_error_get_error_message(const rp_document_error *error,
                                     uint64_t index) {
  revng_check(error != nullptr);
  return error->Reasons.at(index).Message.c_str();
}

static const char *
_rp_document_error_get_error_location(const rp_document_error *error,
                                      uint64_t index) {
  revng_check(error != nullptr);
  return error->Reasons.at(index).Location.c_str();
}

static const char *
_rp_simple_error_get_error_type(const rp_simple_error *error) {
  revng_check(error != nullptr);
  return error->ErrorType.c_str();
}

static const char *_rp_simple_error_get_message(const rp_simple_error *error) {
  revng_check(error != nullptr);
  return error->Message.c_str();
}

static uint64_t _rp_analysis_get_options_count(const rp_analysis *analysis) {
  return analysis->second->getOptionsNames().size();
}

static char *_rp_analysis_get_option_name(const rp_analysis *analysis,
                                          uint64_t extra_argument_index) {
  auto Names = analysis->second->getOptionsNames();
  if (extra_argument_index < 0
      or static_cast<size_t>(extra_argument_index) >= Names.size())
    return nullptr;
  return copyString(Names[extra_argument_index]);
}

static char *_rp_analysis_get_option_type(const rp_analysis *analysis,
                                          uint64_t extra_argument_index) {
  auto Names = analysis->second->getOptionsTypes();
  if (extra_argument_index < 0
      or static_cast<size_t>(extra_argument_index) >= Names.size())
    return nullptr;
  return copyString(Names[extra_argument_index]);
}

static rp_string_map *_rp_string_map_create() {
  return new rp_string_map();
}

static void _rp_string_map_destroy(rp_string_map *map) {
  delete map;
}

static void
_rp_string_map_insert(rp_string_map *map, const char *key, const char *value) {
  auto Result = map->insert_or_assign(key, value);
  revng_assert(Result.second);
}

static rp_invalidations *_rp_invalidations_create() {
  return new rp_invalidations();
}

static void _rp_invalidations_destroy(rp_invalidations *invalidations) {
  delete invalidations;
}

static char *
_rp_invalidations_serialize(const rp_invalidations *invalidations) {
  std::string Out;

  for (auto &StepPair : *invalidations) {
    for (auto &ContainerPair : StepPair.second) {
      for (auto Target : ContainerPair.second) {
        Out += llvm::join_items('/',
                                StepPair.first(),
                                ContainerPair.first(),
                                Target.serialize())
               + '\n';
      }
    }
  }

  return copyString(Out);
}

static uint64_t _rp_kind_get_defined_location_count(const rp_kind *kind) {
  revng_check(kind != nullptr);
  return kind->definedLocations().size();
}

static const rp_rank *_rp_kind_get_defined_location(const rp_kind *kind,
                                                    uint64_t index) {
  revng_check(kind != nullptr);
  const llvm::ArrayRef<const Rank *> Locations = kind->definedLocations();
  if (index < Locations.size())
    return Locations[index];
  else
    return nullptr;
}

static uint64_t _rp_kind_get_preferred_kind_count(const rp_kind *kind) {
  revng_check(kind != nullptr);
  return kind->preferredKinds().size();
}

static const rp_kind *_rp_kind_get_preferred_kind(const rp_kind *kind,
                                                  uint64_t index) {
  revng_check(kind != nullptr);
  const llvm::ArrayRef<const Kind *> PreferredKinds = kind->preferredKinds();
  if (index < PreferredKinds.size())
    return PreferredKinds[index];
  else
    return nullptr;
}

static uint64_t _rp_buffer_size(const rp_buffer *buffer) {
  return buffer->size();
}

static const char *_rp_buffer_data(const rp_buffer *buffer) {
  return buffer->data();
}

static void _rp_buffer_destroy(rp_buffer *buffer) {
  delete buffer;
}

static rp_container_targets_map *_rp_container_targets_map_create() {
  return new ContainerToTargetsMap();
}

static void _rp_container_targets_map_destroy(rp_container_targets_map *map) {
  revng_check(map != nullptr);
  delete map;
}

static void _rp_container_targets_map_add(rp_container_targets_map *map,
                                          const rp_container *container,
                                          const rp_target *target) {
  revng_check(map != nullptr);
  revng_check(container != nullptr);
  revng_check(target != nullptr);
  (*map)[container->second->name()].push_back(*target);
}

static char *_rp_step_get_component(const rp_step *step) {
  revng_check(step != nullptr);
  return copyString(step->getComponent());
}

// NOLINTEND

// Import the autogenerated wrappers, these will contains calls to the
// `wrap<>()` function defined in `Wrapper.h` that will forward the arguments to
// the function of the same name in this file, with an underscore in front.
// For example: rp_initialize will call
// `wrap<"rp_intialize">(rp_initialize, ...)`
// which in turn will call `_rp_initialize`
#include "revng/PipelineC/Wrappers.h"
