#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "PipelineC.h"

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
 * Some functions can report detailed errors, these accept a rp_error_list as
 * their last parameter, which will contain a list of errors in case the return
 * value of the method is false or nullptr. The rp_error_list_* methods can then
 * be used to inspect the errors provided. In case errors are not of interest, a
 * nullptr can be passed instead and errors will be silently ignored.
 */

/**
 * Must be invoked before any other rp_* function is used, must be invoked
 * exactly once. This will take care of initializing all llvm related stuff
 * needed by the revng pipeline to operate.
 *
 * Do not initialize all stuff on your own.
 *
 * \return true if the initialization was successful.
 */
bool rp_initialize(int argc,
                   char *argv[],
                   int libraries_count,
                   const char *libraries_path[],
                   int signals_to_preserve_count,
                   int signals_to_preserve[]);

/**
 * Should be called on clean exit to clean up all LLVM-related stuff used by
 * revng-pipeline.
 *
 * \return true if the shutdown was successful.
 */
bool rp_shutdown();

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
 * \param pipeline_flags_count size of \p pipeline_flags.
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
rp_manager * /*owning*/ rp_manager_create(uint64_t pipelines_count,
                                          const char *pipelines_path[],
                                          uint64_t pipeline_flags_count,
                                          const char *pipeline_flags[],
                                          const char *execution_directory);

/**
 * Same as like rp_manager_create but, instead of the path to a pipeline YAML
 * file, accepts the strings containing the pipeline directly.
 */
rp_manager * /*owning*/
rp_manager_create_from_string(uint64_t pipelines_count,
                              const char *pipelines[],
                              uint64_t pipeline_flags_count,
                              const char *pipeline_flags[],
                              const char *execution_directory);

/**
 * Exactly like rp_manager_create, except that it is run without loading and
 * saving the state of the pipeline before and after the execution, and that no
 * flags are provided.
 */
rp_manager * /*owning*/ rp_manager_create_memory_only(const char *pipeline_path,
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
 *  If path is not nullptr, serialize to the specified path otherwise serialize
 *  in the manager's execution directory
 *  \return 0 if a error happened, 1 otherwise.
 */
bool rp_manager_save(rp_manager *manager, const char *path);

/**
 * Serialize the pipeline context to the specified directory
 *  \return 0 if a error happened, 1 otherwise.
 */
bool rp_manager_save_context(rp_manager *manager, const char *path);

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
 * \return the serialized string rappresenting a global object
 */
const char * /*owning*/
rp_manager_create_global_copy(rp_manager *manager, const char *global_name);

/**
 * Sets the contents of the specified global
 * \param serialied a c-string representing the serialized new global
 * \param global_name the name of the global
 * \return true on success, false otherwise
 */
bool rp_manager_set_global(rp_manager *manager,
                           const char *serialized,
                           const char *global_name,
                           rp_error_list *error_list);

/**
 * Checks that the serialized global would be correct if set as global_name
 * \param serialied a c-string representing the serialized new global
 * \param global_name the name of the global
 * \return true on success, false otherwise
 */
bool rp_manager_verify_global(rp_manager *manager,
                              const char *serialized,
                              const char *global_name,
                              rp_error_list *error_list);

/**
 * Apply the specified diff to the global
 * \param diff a c-string representing the serialized diff
 * \param global_name the name of the global
 * \return true on success, false otherwise
 */
bool rp_manager_apply_diff(rp_manager *manager,
                           const char *diff,
                           const char *global_name,
                           rp_error_list *error_list);

/**
 * Checks that the specified diff would apply correctly to the global
 * \param diff a c-string representing the serialized diff
 * \param global_name the name of the global
 * \return true on success, false otherwise
 */
bool rp_manager_verify_diff(rp_manager *manager,
                            const char *diff,
                            const char *global_name,
                            rp_error_list *error_list);

/**
 * \return the number of serializable global objects
 */
int rp_manager_get_globals_count(rp_manager *manager);

/**
 * \return the owning pointer to the name of serializable global object
 * with the provided index, nullptr if the index was out of bound.
 */
const char * /*owning*/
rp_manager_get_global_name(rp_manager *manager, int index);

/**
 * \return the kind with the provided name, NULL if no kind had the provided
 *         name.
 */
rp_kind *
rp_manager_get_kind_from_name(rp_manager *manager, const char *kind_name);

/**
 * \return the number of kinds present in the manager.
 */
uint64_t rp_manager_kinds_count(rp_manager *manager);

/**
 * \return the kind with the provided index, or NULL if not such step existed.
 */
rp_kind *rp_manager_get_kind(rp_manager *manager, uint64_t index);

/**
 * Request the production of the provided targets in a particular container.
 *
 * \param tagets_count must be equal to the size of targets.
 *
 * \return 0 if an error was encountered, the serialized container otherwise
 */
const char * /*owning*/ rp_manager_produce_targets(rp_manager *manager,
                                                   uint64_t targets_count,
                                                   rp_target *targets[],
                                                   rp_step *step,
                                                   rp_container *container);

/**
 * Request to run the required analysis
 *
 * \param tagets_count must be equal to the size of targets.
 *
 * \return 0 if an error was encountered, the owning diff map of affected global
 * objects
 */
rp_diff_map * /*owning*/ rp_manager_run_analysis(rp_manager *manager,
                                                 uint64_t targets_count,
                                                 rp_target *targets[],
                                                 const char *step_name,
                                                 const char *analysis_name,
                                                 rp_container *container,
                                                 const rp_string_map *options);

/**
 * Request to run all analyses on all targets
 *
 * \return 0 if an error was encountered, the owning diff map of affected global
 * objects
 */
rp_diff_map * /*owning*/
rp_manager_run_all_analyses(rp_manager *manager, const rp_string_map *options);

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
char * /*owning*/ rp_manager_create_container_path(rp_manager *manager,
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
 *         \p step or nullptr if not present.
 */
rp_container *
rp_step_get_container(rp_step *step, rp_container_identifier *identifier);

/**
 * \return a \p step 's parent, if present
 */
rp_step *rp_step_get_parent(rp_step *step);

/**
 * \return the artifacts kind of the step
 */
rp_kind *rp_step_get_artifacts_kind(rp_step *step);

/**
 * \return the artifacts container of the step
 */
rp_container *rp_step_get_artifacts_container(rp_step *step);

/**
 * \return the artifacts filename to use for a single target
 */
const char *rp_step_get_artifacts_single_target_filename(rp_step *step);

/**
 * \return the number of analysis present in this step
 */
int rp_step_get_analyses_count(rp_step *step);

/**
 * \return the of the analysis in the provided step with the provided index.
 *
 * index must be less than rp_step_get_analyses_count
 */
rp_analysis *rp_step_get_analysis(rp_step *step, int index);

/**
 * \return the name of the analysis
 */
const char *rp_analysis_get_name(rp_analysis *analysis);

/**
 * \return the count of containers used by the provided analysis.
 * is no analysis
 */
int rp_analysis_get_arguments_count(rp_analysis *analysis);

/**
 * \return a owning pointer to the name of the container used as index
 * argument of the analysis of this step.
 *
 * index must be less than rp_step_get_analysis_arguments_count(step)
 */
const char * /*owning*/
rp_analysis_get_argument_name(rp_analysis *analysis, int index);

/**
 * \return the quantity of kinds that can be accepted by a analysis
 */
int rp_analysis_get_argument_acceptable_kinds_count(rp_analysis *analysis,
                                                    int argument_index);

/**
 * \return the ammout of extra arguments of the provided analysis
 */
int rp_analysis_get_options_count(rp_analysis *analysis);

/**
 * \return the name of the extra argument with the provided index. Returns
 * nullptr if extra_argument_index < 0 or extra_argument_index >
 * rp_analysis_get_extra_argument_count(analysis)
 */
const char * /*owning*/
rp_analysis_get_option_name(rp_analysis *analysis, int extra_argument_index);

/**
 * \return the type of the extra argument with the provided index. Returns
 * nullptr if extra_argument_index < 0 or extra_argument_index >
 * rp_analysis_get_extra_argument_count(analysis)
 */
const char * /*owning*/
rp_analysis_get_option_type(rp_analysis *analysis, int extra_argument_index);

/**
 * \return the pointer to a acceptable kind for the container with index
 * argument_index within a analysis, nullptr if kind_index is >= than
 * rp_analysis_argument_acceptable_kinds_count(analysis, argument_index)
 */
const rp_kind *rp_analysis_get_argument_acceptable_kind(rp_analysis *analysis,
                                                        int argument_index,
                                                        int kind_index);

/**
 * Serialize a single pipeline step to the specified directory
 *  \return 0 if a error happened, 1 otherwise.
 */
bool rp_step_save(rp_step *step, const char *path);

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
rp_target * /*owning*/ rp_target_create(rp_kind *kind,
                                        int is_exact,
                                        uint64_t path_components_count,
                                        const char *path_components[]);
/**
 * Deserialize a target from a string, arguments cannot be NULL.
 *
 * \return NULL if \p string is malformed.
 */
rp_target * /*owning*/
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
 * Serializes target into a string.
 *
 * \note The return string is owned by the caller. Destroy with
 *       rp_string_destroy.
 */
char * /*owning*/ rp_target_create_serialized_string(rp_target *target);

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

/**
 * \return true if the provided target is currently cached in the provided
 * container. False otherwise
 */
bool rp_target_is_ready(rp_target *target, rp_container *container);

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

/**
 * \return a \p kind 's parent if present, otherwise nullptr.
 */
rp_kind *rp_kind_get_parent(rp_kind *kind);

/**
 * \return the rank associated with the specified \p Kind
 */
rp_rank *rp_kind_get_rank(rp_kind *kind);

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
 * \return the mime type of \p container
 * \note The returned string must not be freed by the caller.
 */
const char *rp_container_get_mime(rp_container *container);

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
 */
bool rp_manager_container_deserialize(rp_manager *manager,
                                      rp_step *step,
                                      const char *container_name,
                                      const char *content,
                                      uint64_t size);

/**
 * \return the serialized content of the element associated to the provided
 * target,
 *
 * \note Target must be already present in container
 *
 */
const char * /*owning*/
rp_container_extract_one(rp_container *container, rp_target *target);

/** \} */

/**
 *
 * \defgroup rp_diff_map rp_diff_map methods
 * \{
 */

/**
 * frees the provided map
 */
void rp_diff_map_destroy(rp_diff_map *to_free);

/**
 * \return nullptr if global_name did not named a global variable in the
 * diff_map else return the serialized diff of the indicated global
 */
const char * /*owning*/
rp_diff_map_get_diff(rp_diff_map *map, const char *global_name);

/**
 * \return true if the rp_diff_map is empty (no changes), false otherwise
 */
bool rp_diff_map_is_empty(rp_diff_map *map);

/** \} */

/**
 * \defgroup rp_rank rp_rank methods
 * \{
 */

/**
 * \return the number of ranks present in the manager
 */
uint64_t rp_ranks_count();

/**
 * \return the rank with the provided index, or NULL if no such rank existed.
 */
rp_rank *rp_rank_get(uint64_t index);

/**
 * \return the rank with the provided name, NULL if no rank had the provided
 *         name.
 */
rp_rank *rp_rank_get_from_name(const char *rank_name);

/**
 * \return the name of \p Rank
 * \note The returned string must not be freed by the caller.
 */
const char *rp_rank_get_name(rp_rank *rank);

/**
 * \return the depth of \p Rank
 */
uint64_t rp_rank_get_depth(rp_rank *rank);

/**
 * \return \p Rank 's parent, or NULL if it has none
 */
rp_rank *rp_rank_get_parent(rp_rank *rank);

/**
 * \defgroup rp_error_list rp_error_list methods
 * \{
 */

/**
 * \return a new rp_error_list
 */
rp_error_list * /*owning*/ rp_make_error_list();

/**
 * \return if an error_list is empty (contains no errors)
 */
bool rp_error_list_is_empty(rp_error_list *error);

/**
 * \return number of errors present
 */
uint64_t rp_error_list_size(rp_error_list *error);

/**
 * \return the error's message at the specified index if present or nullptr
 */
const char * /*owning*/
rp_error_list_get_error_message(rp_error_list *error, uint64_t index);

/**
 * Frees the provided error_error_list
 */
void rp_error_list_destroy(rp_error_list *error);

/** \} */

/**
 * \defgroup rp_string_map rp_string_map methods
 * \{
 */

/**
 * \return a owning pointer to a new string map
 */
rp_string_map * /*owning*/ rp_string_map_create();

/**
 * destroys the provided map
 */
void rp_string_map_destroy(rp_string_map *map);

/**
 * inserts the pair of key and value in the provided map
 */
void rp_string_map_insert(rp_string_map *map,
                          const char *key,
                          const char *value);

/** \} */
