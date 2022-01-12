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
 * rp_TYPE_destroy.
 */

/**
 * Must be invoked before any other rp_* function is used,
 * must be invoked exactly once.
 * This will take care of initializing all llvm related
 * stuff needed by the revng pipeline to operate.
 *
 * Do not initialize all stuff on your own.
 */
bool rp_initialize(int argc,
                   char *argv[],
                   int libraries_count,
                   const char *libraries_path[]);

/**
 * @defgroup rp_targets_list rp container status methods
 */

/**
 * @defgroup rp_container_identifier rp container methods
 */

/**
 * @defgroup rp_step rp step methods
 */

/**
 * @defgroup rp_target rp target methods
 */

/**
 * @defgroup rp_kind rp kind methods
 */

/**
 * @defgroup rp_manager rp manager methods
 * @{
 */

/**
 * loads and setups everything needed to run the pipeline, operating on the
 * provided directory and created from the provided pipelines_path.
 *
 * pipeline_path, flags, execution_directory cannot be null.
 * \param execution_directory can be empty, if it is empty then the content of
 * the pipeline will not be loaded and saved on disk before and after the
 * execution. pipeline_flags can be empty, pipeline_flags_count must be the size
 * of the pipeline_flags array.
 *
 * \return the created rp_manager if no error happened, nullptr otherwise.
 *
 * this function can be called only once, since it will take take of the
 * initialization of all dynamically loaded libraries.
 *
 * this must be invoked after rp_initialize and all invocations of
 * rp_load_library_permanently
 */
rp_manager *rp_manager_create(uint64_t pipelines_count,
                              const char *pipelines_path[],
                              uint64_t pipeline_flags_count,
                              const char *pipeline_flags[],
                              const char *execution_directory);

/**
 * \return the number of containers registered in this pipeline
 */
uint64_t rp_manager_containers_count(rp_manager *manager);

/**
 * \return the container at the provided index
 *
 * \param manager cannot be null, index must be less than
 * rp_manager_containers_count(manager)
 */
rp_container_identifier *
rp_manager_get_container_identifier(rp_manager *manager, uint64_t index);

/** @} */

/**
 * @ingroup rp_container
 * \return the name of the nth container registered inside the manager
 */
const char *
rp_container_identifier_get_name(rp_container_identifier *container);

/**
 *
 *  @ingroup rp_manager
 * exactly like rp_manager_create except that instead of the path to a
 * pipeline yaml file this overload accepts the string containing the pipeline
 * directly
 *
 */
rp_manager *rp_manager_create_from_string(uint64_t pipelines_count,
                                          const char *pipelines[],
                                          uint64_t pipeline_flags_count,
                                          const char *pipeline_flags[],
                                          const char *execution_directory);

/**
 *  @ingroup rp_manager
 * Exactly like rp_manager_create, except that it is run without loading and
 * saving the state of the pipeline before and after the execution, and that no
 * flags are provided.
 */
rp_manager *rp_manager_create_memory_only(const char *pipeline_path,
                                          uint64_t flags_count,
                                          const char *flags[]);

/**
 *  @ingroup rp_manager
 *  Triggers the serialization of the pipeline on disk.
 *  \return 0 if a error happened, 1 otherwise.
 *
 *  \param manager cannot be null
 */
bool rp_manager_store_containers(rp_manager *manager);

/**
 *  @ingroup rp_manager
 * deletes the manager object and destroys all the resourced acquired by it.
 */
void rp_manager_destroy(rp_manager *manager);

/**
 *  @ingroup rp_manager
 * \return the amount of steps present in the manager.
 *
 * \param manager cannot be null
 */
uint64_t rp_manager_steps_count(rp_manager *manager);

const inline uint64_t RP_STEP_NOT_FOUND = UINT64_MAX;

/**
 *  @ingroup rp_manager
 * \return the index of the step with the provided name.
 * returns RP_STEP_NOT_FOUND if no step with such name existed.
 *
 * \param runner cannot be null
 */
uint64_t rp_manager_step_name_to_index(rp_manager *runner, const char *name);

/**
 *  @ingroup rp_manager
 * \return the step with the provided index, or null if not such step existed.
 *
 * \param runner cannot be null
 */
rp_step *rp_manager_get_step(rp_manager *runner, uint64_t index);

/**
 * @ingroup rp_step
 * \return the step name
 *
 * \param step cannot be null
 */
const char *rp_step_get_name(rp_step *step);

/**
 *
 * @ingroup rp_step
 * \return the container associated to the provided container identifier,
 *
 * \param step cannot be null
 *
 */
rp_container *
rp_step_get_container(rp_step *step, rp_container_identifier *identifier);

rp_container *rp_step_get_container_from_name(rp_step *step, const char *name);

/**
 * \return the name of a container
 */
const char *rp_container_get_name(rp_container *container);

/**
 * @ingroup rp_manager
 * \return the kind with the provided name, null if no kind
 * had the provided name.
 *
 * \param runner cannot be null
 * \param kind_name cannot be null
 */
rp_kind *
rp_manager_get_kind_from_name(rp_manager *runner, const char *kind_name);

/**
 *
 * @ingroup rp_kind
 * \return the name of a kind
 *
 * \param kind cannot be null
 */
const char *rp_kind_get_name(rp_kind *kind);

/**
 * @ingroup rp_target
 * creates a target from the provided info
 *
 * \param kind cannot be null
 * \param path_components_count must be equal to the granularity depth of the
 * kind \param path_components size must be equal to path_components_count
 *
 * \return 0 if a error was encountered, 1 otherwise.
 */
rp_target *rp_create_target(rp_kind *kind,
                            int is_exact,
                            uint64_t path_components_count,
                            const char *path_components[]);

/**
 * @ingroup rp_target
 * Deletes the provided target.
 * \note Warning: this target must be one obtained from rp_create_target
 */
void rp_destroy_target(rp_target *target);

/**
 *  @ingroup rp_manager
 * requests the production of the provided targets in a particular container
 *
 * \return 0 if an error was encountered, 1 otherwise
 *
 * \param manager cannot be null
 * \param tagets_count must be equal to the size of targets
 * \param step cannot be null
 * \param container cannot be null
 */
bool rp_manager_produce_targets(rp_manager *manager,
                                uint64_t targets_count,
                                rp_target *targets[],
                                rp_step *step,
                                rp_container *container);

/**
 * serialize the provided container at the provided step in the provided path
 *
 * no argument can be null
 *
 * \return 0 if an error was encountered 1 otherwise
 */
bool rp_container_store(rp_container *container, const char *path);

/**
 * load the provided container at the provided step from the provided path
 *
 * no argument can be null
 *
 * \return 0 if a error was encountered 1 otherwise
 *
 */
bool rp_container_load(rp_container *container, const char *path);

/**
 *  @ingroup rp_manager
 * \return a table containing all the scalar available target that can
 * be requested to the manager
 *
 * \param manager cannot be null
 */
void rp_manager_recompute_all_available_targets(rp_manager *manager);

/**
 *  @ingroup rp_manager
 * \return the container status associated to the provided checkpoint
 * or null if no status is associated to the provided checkpoint
 *
 * no argument can be null
 */
rp_targets_list *rp_manager_get_container_targets_list(rp_manager *manager,
                                                       rp_container *container);

/**
 * @ingroup rp_targets_list
 * \return the amount of target inside the provided container statuses
 */
uint64_t rp_targets_list_targets_count(rp_targets_list *status);

/**
 * @ingroup rp_targets_list
 *  \return the nth target inside the provided status or null if it's out of
 * bounds
 *
 *  \param status cannot be null
 */
rp_target *rp_targets_list_get_target(rp_targets_list *status, uint64_t index);

/**
 * @ingroup rp_target
 * \return the kind of the provided target
 *
 * \param target cannot be null
 */
rp_kind *rp_target_get_kind(rp_target *target);

/**
 * @ingroup rp_target
 *
 * Serializes target into a string, target cannot be null.
 *
 */
char *rp_create_target_serialize(rp_target *target);

/**
 *
 * @ingroup rp_target
 * destroyes a serialized target
 *
 **/
void rp_destroy_serialized_target(const char *serialized_target);

/**
 * @ingroup rp_target
 *
 * Deserialize a target from a string, arguments cannot be null.
 *
 * Returns null if the input string was malformed.
 *
 */
rp_target *
rp_create_target_deserialize(rp_manager *manager, const char *serialized);

/**
 * @ingroup rp_target
 * \return 1 if target is requiring exactly a particular kind
 * \return 0 if target can accept derived from targets
 *
 * \param target cannot be null
 */
bool rp_target_is_exact(rp_target *target);

/**
 * @ingroup rp_target
 * \return the number of path component inside the target
 *
 * \param target cannot be null
 */
uint64_t rp_target_path_components_count(rp_target *target);

/**
 * @ingroup rp_target
 * \return the name of the nth path_component, or "*" if it represents all
 * possible targets. returns null if index is out of bound
 *
 * \param target cannot be null
 */
const char *rp_target_get_path_component(rp_target *target, uint64_t index);

/**
 *  @ingroup rp_manager
 * \return the path of the provided container at the provided step
 */
const char *rp_create_container_path(rp_manager *manager,
                                     const char *step_name,
                                     const char *container_name);

/**
 *  @ingroup rp_manager
 * frees the string returned by rp_create_container_path
 */
void rp_destroy_container_path(const char *container_path);

#ifdef __cplusplus
} // extern C
#endif
