//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"
#include "revng/PipelineC/PipelineC.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/ModelInvalidationEvent.h"
#include "revng/Pipes/PipelineManager.h"
#include "revng/Support/Assert.h"
#include "revng/TupleTree/TupleTreeDiff.h"

#include "sys/types.h"

using namespace pipeline;
using namespace revng::pipes;

static char *copy_string(llvm::StringRef str) {
  char *ToReturn = (char *) malloc(sizeof(char) * (str.size() + 1));
  strncpy(ToReturn, str.data(), str.size());
  ToReturn[str.size()] = 0;
  return ToReturn;
}

static bool Initialized = false;

static bool loadLibraryPermanently(const char *LibraryPath) {
  revng_check(not Initialized);
  revng_check(LibraryPath != nullptr);

  std::string Msg;
  return llvm::sys::DynamicLibrary::LoadLibraryPermanently(LibraryPath, &Msg);
}

bool rp_is_initialized() {
  return Initialized;
}

bool rp_initialize(int argc,
                   char *argv[],
                   int libraries_count,
                   const char *libraries_path[]) {
  if (Initialized)
    return false;

  llvm::cl::ParseCommandLineOptions(argc, argv);
  for (int I = 0; I < libraries_count; I++)
    if (loadLibraryPermanently(libraries_path[I]))
      return false;

  Initialized = true;

  Registry::runAllInitializationRoutines();
  return true;
}

static rp_manager *rp_manager_create_impl(uint64_t pipelines_count,
                                          const char *pipelines_path[],
                                          uint64_t pipeline_flags_count,
                                          const char *pipeline_flags[],
                                          const char *execution_directory,
                                          bool is_path = true) {
  revng_check(Initialized);
  revng_check(pipelines_path != nullptr);
  if (pipeline_flags == nullptr)
    revng_check(pipeline_flags_count == 0);
  revng_check(execution_directory != nullptr);

  std::vector<std::string> FlagsVector;
  for (size_t I = 0; I < pipeline_flags_count; I++)
    FlagsVector.push_back(pipeline_flags[I]);

  std::vector<std::string> Pipelines;
  for (size_t I = 0; I < pipelines_count; I++)
    Pipelines.push_back(pipelines_path[I]);
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

  return new PipelineManager(std::move(*Pipeline));
}

rp_manager *rp_manager_create_from_string(uint64_t pipelines_count,
                                          const char *pipelines[],
                                          uint64_t pipeline_flags_count,
                                          const char *pipeline_flags[],
                                          const char *execution_directory) {
  return rp_manager_create_impl(pipelines_count,
                                pipelines,
                                pipeline_flags_count,
                                pipeline_flags,
                                execution_directory,
                                false);
}

rp_manager *rp_manager_create(uint64_t pipelines_count,
                              const char *pipelines_path[],
                              uint64_t pipeline_flags_count,
                              const char *pipeline_flags[],
                              const char *execution_directory) {
  return rp_manager_create_impl(pipelines_count,
                                pipelines_path,
                                pipeline_flags_count,
                                pipeline_flags,
                                execution_directory);
}

rp_manager *rp_manager_create_memory_only(const char *pipeline_path,
                                          uint64_t flags_count,
                                          const char *flags[]) {
  const char *paths[1] = { pipeline_path };
  return rp_manager_create(1, paths, flags_count, flags, "");
}

bool rp_manager_store_containers(rp_manager *manager) {
  revng_check(manager != nullptr);

  auto Error = manager->storeToDisk();
  if (not Error)
    return true;

  llvm::consumeError(std::move(Error));
  return false;
}

void rp_manager_destroy(rp_manager *manager) {
  revng_check(manager != nullptr);
  delete manager;
}

uint64_t rp_manager_steps_count(rp_manager *manager) {
  revng_check(manager != nullptr);
  return manager->getRunner().size();
}

uint64_t rp_manager_step_name_to_index(rp_manager *manager, const char *name) {
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
rp_step *rp_manager_get_step(rp_manager *manager, uint64_t index) {
  revng_check(manager != nullptr);
  revng_check(index < manager->getRunner().size());
  return &(*(std::next(manager->getRunner().begin(), index)));
}

const char *rp_step_get_name(rp_step *step) {
  revng_check(step != nullptr);
  return step->getName().data();
}

rp_container *
rp_step_get_container(rp_step *step, rp_container_identifier *container) {
  revng_check(step != nullptr);
  revng_check(container != nullptr);
  step->containers()[container->first()];
  auto Iter = step->containers().find(container->first());
  if (Iter == step->containers().end())
    return nullptr;

  return &*Iter;
}

uint64_t rp_targets_list_targets_count(rp_targets_list *targets_list) {
  revng_check(targets_list != nullptr);
  return targets_list->size();
}

rp_kind *
rp_manager_get_kind_from_name(rp_manager *manager, const char *kind_name) {
  revng_check(manager != nullptr);
  revng_check(kind_name != nullptr);

  return manager->getKind(kind_name);
}

const char *rp_kind_get_name(rp_kind *kind) {
  revng_check(kind != nullptr);
  return kind->name().data();
}

rp_kind *rp_kind_get_parent(rp_kind *kind) {
  revng_check(kind != nullptr);
  return kind->parent();
}

const char *rp_manager_produce_targets(rp_manager *manager,
                                       uint64_t targets_count,
                                       rp_target *targets[],
                                       rp_step *step,
                                       rp_container *container) {
  revng_check(manager != nullptr);
  revng_check(targets_count != 0);
  revng_check(targets != nullptr);
  revng_check(step != nullptr);
  revng_check(container != nullptr);

  ContainerToTargetsMap Targets;
  for (size_t I = 0; I < targets_count; I++)
    Targets[container->second->name()].push_back(*targets[I]);

  auto Error = manager->getRunner().run(step->getName(), Targets);
  if (Error) {
    llvm::consumeError(std::move(Error));
    return nullptr;
  }

  std::string Out;
  llvm::raw_string_ostream Serialized(Out);
  const auto &ToFilter = Targets[container->second->name()];
  const auto &Cloned = container->second->cloneFiltered(ToFilter);
  llvm::cantFail(Cloned->serialize(Serialized));
  Serialized.flush();

  return copy_string(Out);
}

rp_target *rp_target_create(rp_kind *kind,
                            int is_exact,
                            uint64_t path_components_count,
                            const char *path_components[]) {
  revng_check(kind != nullptr);
  revng_check(path_components_count != 0);
  revng_check(path_components != nullptr);
  PathComponents List;
  for (size_t I = 0; I < path_components_count; I++) {

    llvm::StringRef Rank(path_components[I]);
    List.push_back(Rank == "*" ? PathComponent::all() :
                                 PathComponent(Rank.str()));
  }

  auto Exact = is_exact ? Exactness::Exact : Exactness::DerivedFrom;
  return new Target(std::move(List), *kind, Exact);
}

void rp_target_destroy(rp_target *target) {
  revng_check(target != nullptr);
  delete target;
}

uint64_t rp_manager_kinds_count(rp_manager *manager) {
  revng_check(manager != nullptr);
  return manager->getRunner().getKindsRegistry().size();
}

rp_kind *rp_manager_get_kind(rp_manager *manager, uint64_t index) {
  revng_check(manager != nullptr);
  revng_check(index < manager->getRunner().getKindsRegistry().size());
  return &*std::next(manager->getRunner().getKindsRegistry().begin(), index);
}

bool rp_container_store(rp_container *container, const char *path) {
  revng_check(container != nullptr);
  revng_check(path != nullptr);
  auto Error = container->second->storeToDisk(path);
  if (not Error)
    return true;

  llvm::consumeError(std::move(Error));
  return false;
}

bool rp_container_load(rp_container *container, const char *path) {
  revng_check(container != nullptr);
  revng_check(path != nullptr);
  auto Error = container->second->loadFromDisk(path);
  if (not Error)
    return true;

  llvm::consumeError(std::move(Error));
  return false;
}

rp_kind *rp_target_get_kind(rp_target *target) {
  revng_check(target != nullptr);
  return &target->getKind();
}

bool rp_target_is_exact(rp_target *target) {
  revng_check(target != nullptr);
  return target->kindExactness() == Exactness::Exact;
}
uint64_t rp_target_path_components_count(rp_target *target) {
  revng_check(target != nullptr);
  return target->getPathComponents().size();
}
const char *rp_target_get_path_component(rp_target *target, uint64_t index) {
  revng_check(target != nullptr);
  auto &PathComponents = target->getPathComponents();
  revng_check(index < PathComponents.size());

  if (PathComponents[index].isAll())
    return "*";

  return PathComponents[index].getName().c_str();
}

char *rp_manager_create_container_path(rp_manager *manager,
                                       const char *step_name,
                                       const char *container_name) {
  revng_check(manager != nullptr);
  revng_check(step_name != nullptr);
  revng_check(container_name != nullptr);

  if (manager->executionDirectory().empty())
    return nullptr;

  auto Path = (manager->executionDirectory() + "/" + step_name + "/"
               + container_name)
                .str();
  return copy_string(Path);
}

void rp_manager_recompute_all_available_targets(rp_manager *manager) {
  revng_check(manager != nullptr);
  manager->recalculateAllPossibleTargets();
}

rp_targets_list *
rp_manager_get_container_targets_list(rp_manager *manager,
                                      rp_container *container) {
  revng_check(manager != nullptr);
  revng_check(container != nullptr);

  return manager->getTargetsAvailableFor(*container);
}

rp_target *
rp_targets_list_get_target(rp_targets_list *targets_list, uint64_t index) {
  revng_check(targets_list != nullptr);
  revng_check(index < targets_list->size());
  return &(*targets_list)[index];
}

void rp_string_destroy(char *string) {
  revng_check(string != nullptr);
  free(string);
}

uint64_t rp_manager_containers_count(rp_manager *manager) {
  revng_check(manager != nullptr);
  return manager->getRunner().registeredContainersCount();
}

rp_container_identifier *
rp_manager_get_container_identifier(rp_manager *manager, uint64_t index) {
  revng_check(manager != nullptr);
  const auto &ContainerRegistry = manager->getRunner().getContainerFactorySet();
  revng_check(index < ContainerRegistry.size());
  return &*std::next(ContainerRegistry.begin(), index);
}

const char *rp_container_get_name(rp_container *container) {
  revng_check(container != nullptr);
  return container->second->name().data();
}

const char *
rp_container_identifier_get_name(rp_container_identifier *identifier) {
  revng_check(identifier != nullptr);
  return identifier->first().data();
}

char *rp_target_create_serialized_string(rp_target *target) {
  revng_check(target != nullptr);
  return copy_string(target->serialize());
}

rp_target *
rp_target_create_from_string(rp_manager *manager, const char *string) {
  revng_check(manager != nullptr);
  revng_check(string != nullptr);
  auto MaybeTarget = parseTarget(string, manager->context().getKindsRegistry());
  if (MaybeTarget)
    return new Target(std::move(*MaybeTarget));

  llvm::consumeError(MaybeTarget.takeError());
  return nullptr;
}

void rp_apply_model_diff(rp_manager *manager, const rp_model_diff *diff) {
  ModelInvalidationEvent Event(*diff);
  llvm::cantFail(Event.run(manager->getRunner()));

  model::Binary &Model(*getWritableModelFromContext(manager->context()));
  diff->apply(Model);
}

const char *rp_manager_create_serialized_global(rp_manager *manager,
                                                const char *global_name) {
  std::string Out;
  llvm::raw_string_ostream Serialized(Out);
  if (auto Error = manager->context().serializeGlobal(Serialized, global_name);
      Error) {
    llvm::consumeError(std::move(Error));
    return nullptr;
  }
  Serialized.flush();
  return copy_string(Out);
}

bool rp_manager_deserialize_global(rp_manager *manager,
                                   const char *serialized,
                                   const char *global_name) {
  auto MaybeBuffer = llvm::MemoryBuffer::getMemBuffer(serialized);
  if (MaybeBuffer == nullptr)
    return false;

  if (auto Error = manager->context().deserializeGlobal(*MaybeBuffer,
                                                        global_name);
      Error) {
    llvm::consumeError(std::move(Error));
    return false;
  }

  return true;
}

uint64_t rp_ranks_count() {
  return Rank::getAll().size();
}

rp_rank *rp_rank_get(uint64_t index) {
  revng_check(index < Rank::getAll().size());
  return Rank::getAll()[index];
}

rp_rank *rp_rank_get_from_name(const char *rank_name) {
  revng_check(rank_name != nullptr);
  for (auto rank : Rank::getAll()) {
    if (rank->name() == rank_name) {
      return rank;
    }
  }
  return nullptr;
}

const char *rp_rank_get_name(rp_rank *rank) {
  revng_check(rank != nullptr);
  return rank->name().data();
}

uint64_t rp_rank_get_depth(rp_rank *rank) {
  revng_check(rank != nullptr);
  return rank->depth();
}

rp_rank *rp_rank_get_parent(rp_rank *rank) {
  revng_check(rank != nullptr);
  return rank->parent();
}

rp_rank *rp_kind_get_rank(rp_kind *kind) {
  revng_check(kind != nullptr);
  return &kind->rank();
}

rp_step *rp_step_get_parent(rp_step *step) {
  revng_check(step != nullptr);
  if (step->hasPredecessor()) {
    return &step->getPredecessor();
  } else {
    return nullptr;
  }
}

const char *rp_container_get_mime(rp_container *container) {
  revng_check(container != nullptr);
  return container->getValue()->mime().c_str();
}
