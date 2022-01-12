#pragma once
//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifdef __cplusplus
#include <cstdint>

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Step.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/PipelineManager.h"
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef __cplusplus
typedef revng::pipes::PipelineManager rp_manager;
#else
typedef struct rp_manager rp_manager;
#endif

#ifdef __cplusplus
typedef const pipeline::Kind rp_kind;
#else
typedef struct rp_kind rp_kind;
#endif

#ifdef __cplusplus
typedef pipeline::Step rp_step;
#else
typedef struct rp_step rp_step;
#endif

#ifdef __cplusplus
typedef pipeline::ContainerSet::value_type rp_container;
#else
typedef struct rp_container rp_container;
#endif

#ifdef __cplusplus
typedef const pipeline::ContainerFactorySet::value_type rp_container_identifier;
#else
typedef struct rp_container_identifier rp_container_identifier;
#endif

#ifdef __cplusplus
typedef const pipeline::Target rp_target;
#else
typedef struct rp_target rp_target;
#endif

#ifdef __cplusplus
typedef const pipeline::TargetsList rp_targets_list;
#else
typedef struct rp_targets_list rp_targets_list;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Every pointer returned by this library is owned by the library and must not
 * be deleted, except for pointers returned by functions containing the word
 * "create". Objects returned by functions named "create" are owned by the
 * caller and must be destroyed invoking the appropriate function called
 * rp_TYPE_destroy, or, in the case of strings, rp_string_destroy.
 *
 * Unless otherwise noted, no pointer arguments can be NULL.
 */

/**
 * Must be invoked before any other rp_* function is used, must be invoked
 * exactly once. This will take care of initializing all llvm related stuff
 * needed by the revng pipeline to operate.
 *
 * Do not initialize all stuff on your own.
 */
bool rp_initialize(int argc,
                   char *argv[],
                   int libraries_count,
                   const char *libraries_path[]);

/**
 * Free a string return by a rp_*_create_* method.
 */
void rp_string_destroy(char *string);

/**
 * \defgroup rp_manager rp_manager methods
 * \{
 */

/**
 * Load and setup everything needed to run the pipeline, operating on the
 * provided directory and created from the provided pipelines_path.
 *
 * \param pipelines_count size of \p pipelines_path.
 * \param pipelines_path cannot be NULL.
 * \param pipeline_flags_count size of \p pipeline_flags.
 * \param pipeline_flags cannot be NULL.
 * \param execution_directory can be empty (but not NULL), if it is empty then
 * the content of the pipeline will not be loaded and saved on disk before and
 * after the execution. pipeline_flags can be empty, pipeline_flags_count must
 * be the size of the pipeline_flags array.
 *
 * \return the created rp_manager if no error happened, NULL otherwise.
 *
 * This function can be called only once, since it will take take of the
 * initialization of all dynamically loaded libraries.
 */
rp_manager *rp_manager_create(uint64_t pipelines_count,
                              const char *pipelines_path[],
                              uint64_t pipeline_flags_count,
                              const char *pipeline_flags[],
                              const char *execution_directory);

/**
 * Same as like rp_manager_create but, instead of the path to a pipeline YAML
 * file, accepts the strings containing the pipeline directly.
 */
rp_manager *rp_manager_create_from_string(uint64_t pipelines_count,
                                          const char *pipelines[],
                                          uint64_t pipeline_flags_count,
                                          const char *pipeline_flags[],
                                          const char *execution_directory);

/**
 * Exactly like rp_manager_create, except that it is run without loading and
 * saving the state of the pipeline before and after the execution, and that no
 * flags are provided.
 */
rp_manager *rp_manager_create_memory_only(const char *pipeline_path,
                                          uint64_t flags_count,
                                          const char *flags[]);

/**
 * Delete the manager object and destroy all the resourced acquired by it.
 */
void rp_manager_destroy(rp_manager *manager);

/**
 * \return the number of containers registered in this pipeline.
 */
uint64_t rp_manager_containers_count(rp_manager *manager);

/**
 * \param index must be less than rp_manager_containers_count(manager).
 *
 * \return the container at the provided index.
 */
rp_container_identifier *
rp_manager_get_container_identifier(rp_manager *manager, uint64_t index);

/**
 *  Trigger the serialization of the pipeline on disk.
 *
 *  \return 0 if a error happened, 1 otherwise.
 */
bool rp_manager_store_containers(rp_manager *manager);

/**
 * \return the number of steps present in the manager.
 */
uint64_t rp_manager_steps_count(rp_manager *manager);

const inline uint64_t RP_STEP_NOT_FOUND = UINT64_MAX;

/**
 * \return the index of the step with the given name or RP_STEP_NOT_FOUND if no
 *         step with such name was found.
 */
uint64_t rp_manager_step_name_to_index(rp_manager *manager, const char *name);

/**
 * \return the step with the provided index, or NULL if not such step existed.
 */
rp_step *rp_manager_get_step(rp_manager *manager, uint64_t index);

/**
 * \return the kind with the provided name, NULL if no kind had the provided
 *         name.
 */
rp_kind *
rp_manager_get_kind_from_name(rp_manager *manager, const char *kind_name);

/**
 * Request the production of the provided targets in a particular container.
 *
 * \param tagets_count must be equal to the size of targets.
 *
 * \return 0 if an error was encountered, 1 otherwise.
 */
bool rp_manager_produce_targets(rp_manager *manager,
                                uint64_t targets_count,
                                rp_target *targets[],
                                rp_step *step,
                                rp_container *container);

/**
 *
 */
void rp_manager_recompute_all_available_targets(rp_manager *manager);

/**
 * \return the container status associated to the provided \p container
 *         or NULL if no status is associated to the provided container.
 */
rp_targets_list *rp_manager_get_container_targets_list(rp_manager *manager,
                                                       rp_container *container);

/**
 * \return the path of the provided container at the provided step.
 *
 * \note The returned string is owned by the caller. Destroy with
 *       rp_string_destroy.
 */
char *rp_manager_create_container_path(rp_manager *manager,
                                       const char *step_name,
                                       const char *container_name);

/** \} */

/**
 * \defgroup rp_targets_list rp_targets_list methods
 * \{
 */

/**
 * \return the number of targets in the provided container statuses.
 */
uint64_t rp_targets_list_targets_count(rp_targets_list *targets_list);

// HERE
/**
 * \return the n-th target inside the provided \p targets_list or NULL if it's
 *         out of bounds.
 */
rp_target *
rp_targets_list_get_target(rp_targets_list *targets_list, uint64_t index);

/** \} */

/**
 * \defgroup rp_container_identifier rp_container_identifier methods
 * \{
 */

/**
 * \return the name of the n-th container registered inside the manager.
 *
 * \note The returned string must not be freed by the caller.
 */
const char *
rp_container_identifier_get_name(rp_container_identifier *container_identifier);

/** \} */

/**
 * \defgroup rp_step rp_step methods
 * \{
 */

/**
 * \return the step name.
 */
const char *rp_step_get_name(rp_step *step);

/**
 * \return the container associated to the provided \p identifier at the given
 *         \p step.
 */
rp_container *
rp_step_get_container(rp_step *step, rp_container_identifier *identifier);

/** \} */

/**
 * \defgroup rp_target rp_target methods
 * \{
 */

/**
 * Create a target from the provided info.
 *
 * \param path_components_count must be equal to the rank depth of the
 *                              kind.
 *
 * \return 0 if a error was encountered, 1 otherwise.
 */
rp_target *rp_target_create(rp_kind *kind,
                            int is_exact,
                            uint64_t path_components_count,
                            const char *path_components[]);
/**
 * Deserialize a target from a string, arguments cannot be NULL.
 *
 * \return NULL if \p string is malformed.
 */
rp_target *
rp_target_create_from_string(rp_manager *manager, const char *string);

/**
 * Delete the provided target.
 *
 * \note Call this method *only* on rp_target instance created by
 *       rp_create_target.
 */
void rp_target_destroy(rp_target *target);

/**
 * \return the kind of the provided target
 */
rp_kind *rp_target_get_kind(rp_target *target);

/**
 * Serializes target into a string, target cannot be NULL.
 *
 * \note The return string is owned by the caller. Destroy with
 *       rp_string_destroy.
 */
char *rp_target_create_serialized_string(rp_target *target);

/**
 * \return 1 if \p target is requiring exactly a particular kind, 0 if target
 *         can accept derived from targets.
 */
bool rp_target_is_exact(rp_target *target);

/**
 * \return the number of path component in \p target.
 */
uint64_t rp_target_path_components_count(rp_target *target);

/**
 * \return the name of the n-th path_component, or "*" if it represents all
 *         possible targets. Returns NULL if \p index is out of bound.
 *
 * \note The returned string must not be freed by the caller.
 */
const char *rp_target_get_path_component(rp_target *target, uint64_t index);

/** \} */

/**
 * \defgroup rp_kind rp_kind methods
 * \{
 */

/**
 *
 * \return the name of \p kind.
 *
 * \note The returned string must not be freed by the caller.
 */
const char *rp_kind_get_name(rp_kind *kind);

/** \} */

/**
 * \defgroup rp_container rp_container methods
 * \{
 */

/**
 * \return the name of \p container.
 *
 * \note The returned string must not be freed by the caller.
 */
const char *rp_container_get_name(rp_container *container);

/**
 * Serialize \p container in \p path.
 *
 * \return 0 if an error was encountered 1 otherwise.
 */
bool rp_container_store(rp_container *container, const char *path);

/**
 * Load the provided container from the provided path.
 *
 * \return 0 if a error was encountered 1 otherwise
 *
 */
bool rp_container_load(rp_container *container, const char *path);

/** \} */

#ifdef __cplusplus
} // extern C
#endif
