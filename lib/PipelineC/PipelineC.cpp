//
// This file is distributed under the MIT License. See LICENSE.md for details.
//
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/Registry.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"
#include "revng/PipelineC/PipelineC.h"
#include "revng/Pipes/PipelineManager.h"
#include "revng/Support/Assert.h"

using namespace pipeline;
using namespace revng::pipes;

static bool Initialized = false;

static bool rp_load_library_permanently(const char *library_path) {
  revng_assert(not Initialized);
  if (library_path == nullptr)
    return true;

  std::string Msg;
  return not llvm::sys::DynamicLibrary::LoadLibraryPermanently(library_path,
                                                               &Msg);
}

bool rp_initialize(int argc,
                   char *argv[],
                   int libraries_count,
                   const char *libraries_path[]) {
  if (Initialized)
    return false;

  Initialized = true;
  llvm::cl::ParseCommandLineOptions(argc, argv);
  for (int i = 0; i < libraries_count; i++)
    if (not rp_load_library_permanently(libraries_path[i]))
      return false;

  Registry::runAllInitializationRoutines();
  return true;
}

static rp_manager *rp_manager_create_impl(uint64_t pipelines_count,
                                          const char *pipelines_path[],
                                          uint64_t pipeline_flags_count,
                                          const char *pipeline_flags[],
                                          const char *execution_directory,
                                          bool is_path = true) {
  revng_assert(Initialized);
  std::vector<std::string> FlagsVector;
  for (size_t I = 0; I < pipeline_flags_count; I++)
    FlagsVector.push_back(pipeline_flags[I]);

  auto Directory = execution_directory != nullptr ? execution_directory : "";

  std::vector<std::string> Pipelines;
  for (size_t I = 0; I < pipelines_count; I++)
    Pipelines.push_back(pipelines_path[I]);
  auto Pipeline = is_path ?
                    PipelineManager::create(Pipelines, FlagsVector, Directory) :
                    PipelineManager::createFromMemory(Pipelines,
                                                      FlagsVector,
                                                      Directory);
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
  return rp_manager_create(1, paths, flags_count, flags, nullptr);
}

bool rp_manager_store_containers(rp_manager *manager) {
  auto Error = manager->storeToDisk();
  if (not Error)
    return true;

  llvm::consumeError(std::move(Error));
  return false;
}

void rp_manager_destroy(rp_manager *manager) {
  delete manager;
}

uint64_t rp_manager_steps_count(rp_manager *manager) {
  return manager->getRunner().size();
}

uint64_t rp_manager_step_name_to_index(rp_manager *manager, const char *name) {
  size_t I = 0;
  for (const auto &Step : manager->getRunner()) {
    if (Step.getName() == name)
      return I;
    I++;
  }

  return RP_STEP_NOT_FOUND;
}
rp_step *rp_manager_get_step(rp_manager *manager, uint64_t index) {
  revng_assert(manager->getRunner().size() >= index);
  return &(*(std::next(manager->getRunner().begin(), index)));
}

const char *rp_step_get_name(rp_step *step) {
  return step->getName().data();
}

rp_container *
rp_step_get_container(rp_step *step, rp_container_identifier *container) {
  step->containers()[container->first()];
  auto Iter = step->containers().find(container->first());
  if (Iter == step->containers().end())
    return nullptr;

  return &*Iter;
}

uint64_t rp_targets_list_targets_count(rp_targets_list *status) {
  return status->size();
}

rp_container *rp_step_get_container_from_name(rp_step *step, const char *name) {
  if (not step->containers().containsOrCanCreate(name))
    return nullptr;

  // Causes the container to be materialized
  step->containers()[name];

  return &*step->containers().find(name);
}

rp_kind *
rp_manager_get_kind_from_name(rp_manager *manager, const char *kind_name) {
  revng_assert(manager != nullptr);
  revng_assert(kind_name != nullptr);

  return manager->getKind(kind_name);
}

const char *rp_kind_get_name(rp_kind *kind) {
  return kind->name().data();
}

bool rp_manager_produce_targets(rp_manager *manager,
                                uint64_t targets_count,
                                rp_target *targets[],
                                rp_step *step,
                                rp_container *container) {
  revng_assert(targets_count != 0);
  revng_assert(manager != nullptr);

  ContainerToTargetsMap Targets;
  for (size_t I = 0; I < targets_count; I++)
    Targets[container->second->name()].push_back(*targets[I]);

  auto Error = manager->getRunner().run(step->getName(), Targets);
  if (not Error)
    return true;

  llvm::consumeError(std::move(Error));
  return false;
}

rp_target *rp_create_target(rp_kind *kind,
                            int is_exact,
                            uint64_t path_components_count,
                            const char *path_components[]) {
  revng_assert(kind != nullptr);
  PathComponents List;
  for (size_t I = 0; I < path_components_count; I++) {

    llvm::StringRef Gran(path_components[I]);
    List.push_back(Gran == "*" ? PathComponent::all() :
                                 PathComponent(Gran.str()));
  }

  auto Exact = is_exact ? Exactness::Exact : Exactness::DerivedFrom;
  return new Target(std::move(List), *kind, Exact);
}

void rp_destroy_target(rp_target *target) {
  delete target;
}

bool rp_container_store(rp_container *container, const char *path) {
  revng_assert(container != nullptr);
  revng_assert(path != nullptr);
  auto Error = container->second->storeToDisk(path);
  if (not Error)
    return true;

  llvm::consumeError(std::move(Error));
  return false;
}

bool rp_container_load(rp_container *container, const char *path) {
  revng_assert(container != nullptr);
  revng_assert(path != nullptr);
  auto Error = container->second->loadFromDisk(path);
  if (not Error)
    return true;

  llvm::consumeError(std::move(Error));
  return false;
}

rp_kind *rp_target_get_kind(rp_target *target) {
  return &target->getKind();
}

bool rp_target_is_exact(rp_target *target) {
  return target->kindExactness() == Exactness::Exact;
}
uint64_t rp_target_path_components_count(rp_target *target) {
  return target->getPathComponents().size();
}
const char *rp_target_get_path_component(rp_target *target, uint64_t index) {
  if (target->getPathComponents()[index].isAll())
    return "*";

  return target->getPathComponents()[index].getName().c_str();
}

static char *copy_string(llvm::StringRef str) {
  char *ToReturn = (char *) malloc(sizeof(char) * (str.size() + 1));
  strncpy(ToReturn, str.data(), str.size());
  ToReturn[str.size()] = 0;
  return ToReturn;
}

const char *rp_create_container_path(rp_manager *manager,
                                     const char *step_name,
                                     const char *container_name) {
  if (manager->executionDirectory().empty())
    return nullptr;

  auto Path = (manager->executionDirectory() + "/" + step_name + "/"
               + container_name)
                .str();
  return copy_string(Path);
}

void rp_manager_recompute_all_available_targets(rp_manager *manager) {
  manager->recalculateAllPossibleTargets();
}

rp_targets_list *
rp_manager_get_container_targets_list(rp_manager *manager,
                                      rp_container *container) {
  revng_assert(manager != nullptr);
  revng_assert(container != nullptr);

  return manager->getTargetsAvailableFor(*container);
}

rp_target *rp_targets_list_get_target(rp_targets_list *status, uint64_t index) {
  revng_assert(status != nullptr);
  return &(*status)[index];
}

void rp_destroy_container_path(const char *container_path) {
  free(const_cast<char *>(container_path));
}

uint64_t rp_manager_containers_count(rp_manager *manager) {
  return manager->getRunner().registeredContainersCount();
}

rp_container_identifier *
rp_manager_get_container_identifier(rp_manager *manager, uint64_t index) {
  const auto &ContainerRegistry = manager->getRunner().getContainerFactorySet();
  if (ContainerRegistry.size() <= index)
    return nullptr;
  return &*std::next(ContainerRegistry.begin(), index);
}

const char *rp_container_get_name(rp_container *container) {
  return container->second->name().data();
}

const char *
rp_container_identifier_get_name(rp_container_identifier *container) {
  revng_assert(container != nullptr);
  return container->first().data();
}

char *rp_create_target_serialize(rp_target *target) {
  return copy_string(target->serialize());
}

void rp_destroy_serialized_target(const char *serialized_target) {
  free(const_cast<char *>(serialized_target));
}

rp_target *
rp_create_target_deserialize(rp_manager *manager, const char *serialized) {
  auto MaybeTarget = parseTarget(serialized,
                                 manager->context().getKindsRegistry());
  if (MaybeTarget)
    return new Target(std::move(*MaybeTarget));

  llvm::consumeError(MaybeTarget.takeError());
  return nullptr;
}
