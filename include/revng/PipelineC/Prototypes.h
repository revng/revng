#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipelineC/PipelineC.h"
#include "revng/PipelineC/Tracing/LengthHint.h"

/**
 * TODO: move to mkdocs
 *
 * \page pipelineC_convention PipelineC API conventions
 *
 * \section pipelineC_owning_pointers Owning Pointers
 *
 * Every pointer returned by this library is owned by the library and must not
 * be deleted, except for pointers returned by functions with the \c 'owning'
 * comment in the return type, these are owned by caller and must be destroyed
 * invoking the appropriate function called \a rp_TYPE_destroy, or, in the case
 * of strings, rp_string_destroy().
 *
 * Unless otherwise noted, no pointer arguments can be \c NULL .
 *
 *
 * \section pipelineC_passing_arrays Passing Arrays
 *
 * Sometimes an array needs to be passed to a method, in this case there will be
 * two arguments one with the name \a $ARGUMENT which is an array of pointers to
 * the elements and one with the name \a $ARGUMENT_count that indicates how many
 * elements the array has.
 *
 *
 * \section pipelineC_invalidations Invalidations
 *
 * If a user of the pipeline wishes to cache the output of
 * rp_manager_produce_targets() it can do so, however some modifications can
 * lead to a produced output to be invalid. To help with this there is
 * ::rp_invalidations , which if provided to the methods that have it in their
 * argument, will fill in all the targets that have been invalidated by the
 * operation. If the user is not interested in having invalidation information
 * the parameter can be set to \c NULL .
 *
 *
 * \section pipelineC_error_reporting Error Reporting
 *
 * Some functions can report detailed errors, these accept a ::rp_error as
 * their last parameter. After calling it will contain either an
 * rp_document_error (which can represent multiple locations) or a
 * rp_simple_error (which is a generic error without a location) in case the
 * return value of the method is false or \c NULL . The \a rp_error* methods can
 * then be used to inspect the rp_error and extract the underlying
 * rp_{simple,document}_error object. In case errors are not of interest, a
 * \c NULL can be passed instead and errors will be silently ignored.
 */

/*
 * This API tries to be as const-correct as possible. Currently this is not
 * always respected since the main user of the library (Python's revng.api)
 * uses cffi which does not check constness.
 * A proper solution to ensure this would be, at testing time:
 * 1. Have a $CLANG_MAGIC tool that takes PipelineC.cpp and, for each argument,
 *    tries to add const if absent and see if it compiles
 *    if it does -> throw an error because we forgot `const`
 * 2. Once (1) is done there would be a second "pass" that would look at all the
 *    function's return values and check if there is unnecessary non-constness
 *    e.g. if there's `rp_foo *foo()` but all functions accept as argument
 *         `const rp_foo` then there should be an error, since foo should return
 *         `const foo *`
 */

// NOLINTBEGIN

/**
 * Must be invoked before any other rp_* function is used, must be invoked
 * exactly once. This will take care of initializing shared libraries and all of
 * the llvm related stuff needed by the revng pipeline to operate.
 *
 * \return true if the initialization was successful.
 */
bool rp_initialize(int argc,
                   const char *argv[],
                   uint32_t signals_to_preserve_count,
                   int signals_to_preserve[]);
LENGTH_HINT(rp_initialize, 1, 0)
LENGTH_HINT(rp_initialize, 3, 2)

/**
 * Should be called on clean exit to clean up all LLVM-related stuff used by
 * revng-pipeline.
 *
 * \return true if the shutdown was successful.
 */
bool rp_shutdown();

/**
 * Free a string that was returned as an owning pointer.
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
 * \param pipelines_path path to the pipeline yamls to load.
 * \param pipeline_flags flags to pass to the pipelines.
 * \param execution_directory where the manager will save the state of the
 * pipeline to preserve it across restarts. If set to the empty string the
 * pipeline will be ephemeral and all changes will be lost at shutdown.
 *
 * \return the created rp_manager if no error happened, NULL otherwise.
 */
rp_manager * /*owning*/ rp_manager_create(uint64_t pipeline_flags_count,
                                          const char *pipeline_flags[],
                                          const char *execution_directory);
LENGTH_HINT(rp_manager_create, 1, 0)

/**
 * Takes the same arguments as \related rp_manager_create but, instead of the
 * path to a pipeline YAML file, accepts the strings containing the pipeline
 * directly.
 */
rp_manager * /*owning*/
rp_manager_create_from_string(uint64_t pipelines_count,
                              const char *pipelines[],
                              uint64_t pipeline_flags_count,
                              const char *pipeline_flags[],
                              const char *execution_directory);
LENGTH_HINT(rp_manager_create_from_string, 1, 0)
LENGTH_HINT(rp_manager_create_from_string, 3, 2)

/**
 * Delete the manager object and destroy all the resourced acquired by it.
 */
void rp_manager_destroy(rp_manager *manager);

/**
 * \param name the container name to fetch
 *
 * \return the container with the given name
 */
const rp_container_identifier *
rp_manager_get_container_identifier_from_name(const rp_manager *manager,
                                              const char *name);

/**
 *  Trigger the serialization of the pipeline on disk.
 *  \return false if there was an error while saving, true otherwise
 */
bool rp_manager_save(rp_manager *manager);

/**
 * \return the step with the provided name, or NULL if not such step existed.
 */
rp_step *rp_manager_get_step_from_name(rp_manager *manager, const char *name);

/**
 * \param global_name the name of the global
 * \return the serialized string rappresenting a global object
 */
char * /*owning*/
rp_manager_create_global_copy(const rp_manager *manager,
                              const char *global_name);

/**
 * \return the kind with the provided name, NULL if no kind had the provided
 *         name.
 */
const rp_kind *rp_manager_get_kind_from_name(const rp_manager *manager,
                                             const char *kind_name);

/**
 * Request the production of the provided targets in a particular container.
 *
 * \param tagets_count must be equal to the size of targets.
 *
 * \return 0 if an error was encountered, the serialized container otherwise
 */
rp_buffer * /*owning*/
rp_manager_produce_targets(rp_manager *manager,
                           const rp_step *step,
                           const rp_container *container,
                           uint64_t targets_count,
                           const rp_target *targets[],
                           rp_error *error);
LENGTH_HINT(rp_manager_produce_targets, 4, 3)

/**
 * Request to run the required analysis
 *
 * \param targets the targets which the analysis will be run on
 * \param step_name the name of the step
 * \param analysis_name the name of the analysis in that step
 * \param container the container to operate on
 * \param invalidations see \ref pipelineC_invalidations
 * \param options key-value associative array of options to pass to the analysis
 * This option accepts nullptr in case there are no options to pass
 *
 * \return nullptr if an error was encountered, the owning diff map of affected
 * global objects otherwise
 */
rp_diff_map * /*owning*/
rp_manager_run_analysis(rp_manager *manager,
                        const char *step_name,
                        const char *analysis_name,
                        const rp_container_targets_map *target_map,
                        const rp_string_map *options,
                        rp_invalidations *invalidations,
                        rp_error *error);

/**
 * Request to run all analyses of the given list on all targets
 * \param invalidations see \ref pipelineC_invalidations
 * \param options key-value associative array of options to pass to the analysis
 * This option accepts nullptr in case there are no options to pass
 *
 * \return nullptr if an error was encountered, the owning diff map of affected
 * global objects otherwise
 */
rp_diff_map * /*owning*/
rp_manager_run_analyses_list(rp_manager *manager,
                             const char *list_name,
                             const rp_string_map *options,
                             rp_invalidations *invalidations,
                             rp_error *error);

/**
 * \return the container status associated to the provided \p container
 *         or NULL if no status is associated to the provided container.
 */
const rp_targets_list *
rp_manager_get_container_targets_list(const rp_manager *manager,
                                      const rp_container *container);

/**
 * \return The pipeline description file, in YAML format
 */
const char *rp_manager_get_pipeline_description(rp_manager *manager);

/**
 * Returns the Context Commit Index
 */
uint64_t rp_manager_get_context_commit_index(rp_manager *manager);

/** \} */

/**
 * \defgroup rp_targets_list rp_targets_list methods
 * \{
 */

/**
 * \return the number of targets in the provided container statuses.
 */
uint64_t rp_targets_list_targets_count(const rp_targets_list *targets_list);

/**
 * \return the n-th target inside the provided \p targets_list or NULL if it's
 *         out of bounds.
 */
const rp_target * /* owning */
rp_targets_list_get_target(const rp_targets_list *targets_list, uint64_t index);

/** \} */

/**
 * \defgroup rp_step rp_step methods
 * \{
 */

/**
 * \return the container associated to the provided \p identifier at the given
 *         \p step or nullptr if not present.
 */
rp_container *rp_step_get_container(rp_step *step,
                                    const rp_container_identifier *identifier);

/** \} */

/**
 * \defgroup rp_target rp_target methods
 * \{
 */

/**
 * Create a target from the provided info.
 *
 * \param kind the kind of the target
 * \param is_exact if true only the kind specified is considedered, whereas if
 * false the kind and all its children are considered \param path_components a
 * list of strings containing the component of the target
 *
 * \return 0 if a error was encountered, 1 otherwise.
 */
rp_target * /*owning*/ rp_target_create(const rp_kind *kind,
                                        uint64_t path_components_count,
                                        const char *path_components[]);
LENGTH_HINT(rp_target_create, 2, 1)

/**
 * Delete the provided target.
 */
void rp_target_destroy(rp_target *target);

/**
 * \return the kind of the provided target
 */
const char *rp_target_get_kind(const rp_target *target);

/**
 * Serializes target into a string.
 */
char * /*owning*/ rp_target_create_serialized_string(const rp_target *target);

/**
 * \return the number of path component in \p target.
 */
uint64_t rp_target_path_components_count(const rp_target *target);

/**
 * \return the name of the n-th path_component, or "*" if it represents all
 *         possible targets. Returns NULL if \p index is out of bound.
 *
 * \note The returned string must not be freed by the caller.
 */
const char *rp_target_get_path_component(const rp_target *target,
                                         uint64_t index);

/**
 * \return true if the provided target is currently cached in the provided
 * container. False otherwise
 */
bool rp_target_is_ready(const rp_target *target, const rp_container *container);

/** \} */

/**
 * \defgroup rp_container rp_container methods
 * \{
 */

/**
 * \return the mime type of \p container
 */
const char *rp_container_get_mime(const rp_container *container);

/**
 * Load the provided container given a buffer
 * \param step the step where the container resides
 * \param container_name the name of the container
 * \param content pointer to a byte buffer
 * \param size number of bytes contained in the buffer
 *
 * \return false if a error was encountered, true otherwise
 */
bool rp_manager_container_deserialize(rp_manager *manager,
                                      rp_step *step,
                                      const char *container_name,
                                      const char *content,
                                      uint64_t size,
                                      rp_invalidations *invalidations);
LENGTH_HINT(rp_manager_container_deserialize, 3, 4)

/**
 * \return the serialized content of the element associated to the provided
 * target, or nullptr if the content hasn't been produced yet
 */
rp_buffer * /*owning*/
rp_container_extract_one(const rp_container *container,
                         const rp_target *target);

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
 * \param global_name the name of the global variable
 * \return nullptr if the global is not present in the diff_map, otherwise a
 * string of serialized version of the changes applied to the global
 */
char * /*owning*/
rp_diff_map_get_diff(const rp_diff_map *map, const char *global_name);

/**
 * \return true if the rp_diff_map is empty (no changes), false otherwise
 */
bool rp_diff_map_is_empty(const rp_diff_map *map);

/** \} */

/**
 * \defgroup rp_error rp_error methods
 * \{
 */

/**
 * \return a error containing a success
 */
rp_error * /*owning*/ rp_error_create();

/**
 * \return true if this error encodes the success state.
 */
bool rp_error_is_success(const rp_error *error);

/**
 * \return true if error contains a rp_document_error
 */
bool rp_error_is_document_error(const rp_error *error);

/**
 * \return a valid pointer to a document error if error is a document error,
 * nullptr otherwise
 */
rp_document_error *rp_error_get_document_error(rp_error *error);

/**
 * \return a valid pointer to a simple error if error is a simple error,
 * nullptr otherwise
 */
rp_simple_error *rp_error_get_simple_error(rp_error *error);

/**
 * Frees the provided error_error_list
 */
void rp_error_destroy(rp_error *error);

/** \} */

/**
 * \defgroup rp_document_error rp_document_error methods
 */

/**
 * \return number of errors present
 */
uint64_t rp_document_error_reasons_count(const rp_document_error *error);

/**
 * \return the error's type
 */
const char *rp_document_error_get_error_type(const rp_document_error *error);

/**
 * \return the error's location type
 */
const char *rp_document_error_get_location_type(const rp_document_error *error);

/**
 * \return the error's message at the specified index if present or nullptr
 * otherwise
 */
const char *rp_document_error_get_error_message(const rp_document_error *error,
                                                uint64_t index);

/**
 * \return the error's location at the specified index if present or nullptr
 * otherwise
 */
const char *rp_document_error_get_error_location(const rp_document_error *error,
                                                 uint64_t index);

/** \} */

/**
 * \defgroup rp_document_error rp_document_error methods
 */

/**
 * \return the error's type
 */
const char *rp_simple_error_get_error_type(const rp_simple_error *error);

/**
 * \return the error's message
 */
const char *rp_simple_error_get_message(const rp_simple_error *error);

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

/**
 * \defgroup rp_invalidations rp_invalidations methods
 * \{
 */

/**
 * Create a new rp_invalidations object
 * \return owning pointer to the newly created object
 */
rp_invalidations * /*owning*/ rp_invalidations_create();

/**
 * Free a rp_invalidations
 */
void rp_invalidations_destroy(rp_invalidations *invalidations);

/**
 * \return a string where each line is a target that has been invalidated
 */
char * /* owning */
rp_invalidations_serialize(const rp_invalidations *invalidations);

/** \} */

/**
 * \defgroup rp_buffer rp_buffer methods
 * \{
 */

/**
 * \return the length of the buffer
 */
uint64_t rp_buffer_size(const rp_buffer *buffer);

/**
 * \return the pointer at the start of the buffer
 */
const char *rp_buffer_data(const rp_buffer *buffer);

/**
 * Free a rp_buffer
 */
void rp_buffer_destroy(rp_buffer *buffer);

/** \} */

/**
 * \defgroup rp_container_targets_map rp_container_targets_map methods
 * \{
 */

/**
 * Create a new rp_container_targets_map object
 * \return owning pointer to the newly created object
 */
rp_container_targets_map * /*owning*/ rp_container_targets_map_create();

/**
 * Free a rp_container_targets_map
 */
void rp_container_targets_map_destroy(rp_container_targets_map *map);

/**
 * Add the specified Target to the Container in the map
 */
void rp_container_targets_map_add(rp_container_targets_map *map,
                                  const rp_container *container,
                                  const rp_target *target);

/** \} */

// NOLINTEND
