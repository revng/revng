#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

set(PYTHON_WHL_VERSION_PREFIX 0.1)
execute_process(
  COMMAND
    python3 -c
    "import time; print(f\"${PYTHON_WHL_VERSION_PREFIX}.{int(time.time())}\", end='')"
  OUTPUT_VARIABLE PYTHON_WHL_VERSION)

# ~~~
# Declares a python module as a target
# Named arguments:
# - TARGET_NAME: the name of the CMake target to be created
# - WHEEL: the name of the wheel where the files will be included into
# - MODULE_INIT: if specified, this points to the top-level __init__.py file of
#                the package. This needs to be specified separately because
#                CMake will replaces placeholders such as '@VERSION@' with the
#                correct value
# - MODULE_FILES: list of relative paths of the python module sources
# - MODULE_GENERATED_FILES: list of relative paths of the python module
#                           generated sources
# The module sources will be copied to the build directory
# ~~~
function(python_module)
  set(options)
  set(oneValueArgs TARGET_NAME WHEEL MODULE_INIT)
  set(multiValueArgs MODULE_FILES MODULE_GENERATED_FILES)
  cmake_parse_arguments(PYTHON_MODULE "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  if(NOT "${PYTHON_MODULE_MODULE_INIT}" STREQUAL "")
    execute_process(
      COMMAND sed "s;@VERSION@;${PYTHON_WHL_VERSION};"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      INPUT_FILE "${PYTHON_MODULE_MODULE_INIT}"
      OUTPUT_VARIABLE MODULE_INIT_CONTENTS)
    file(
      WRITE
      "${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}/${PYTHON_MODULE_MODULE_INIT}"
      "${MODULE_INIT_CONTENTS}")
    file(
      WRITE
      "${CMAKE_BINARY_DIR}/build-wheel/${PYTHON_MODULE_WHEEL}/${PYTHON_MODULE_MODULE_INIT}"
      "${MODULE_INIT_CONTENTS}")
    list(
      APPEND
      "WHEEL_FILES_${PYTHON_MODULE_WHEEL}"
      "${CMAKE_BINARY_DIR}/build-wheel/${PYTHON_MODULE_WHEEL}/${PYTHON_MODULE_MODULE_INIT}"
    )
  endif()

  foreach(MODULE_FILE ${PYTHON_MODULE_MODULE_FILES})
    configure_file(
      "${MODULE_FILE}"
      "${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}/${MODULE_FILE}" COPYONLY)
    configure_file(
      "${MODULE_FILE}"
      "${CMAKE_BINARY_DIR}/build-wheel/${PYTHON_MODULE_WHEEL}/${MODULE_FILE}"
      COPYONLY)
    list(
      APPEND "WHEEL_FILES_${PYTHON_MODULE_WHEEL}"
      "${CMAKE_BINARY_DIR}/build-wheel/${PYTHON_MODULE_WHEEL}/${MODULE_FILE}")
  endforeach()

  foreach(MODULE_FILE ${PYTHON_MODULE_MODULE_GENERATED_FILES})
    add_custom_command(
      OUTPUT
        "${CMAKE_BINARY_DIR}/build-wheel/${PYTHON_MODULE_WHEEL}/${MODULE_FILE}"
      DEPENDS "${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}/${MODULE_FILE}"
      COMMAND
        cp -a "${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}/${MODULE_FILE}"
        "${CMAKE_BINARY_DIR}/build-wheel/${PYTHON_MODULE_WHEEL}/${MODULE_FILE}")
    list(
      APPEND "WHEEL_FILES_${PYTHON_MODULE_WHEEL}"
      "${CMAKE_BINARY_DIR}/build-wheel/${PYTHON_MODULE_WHEEL}/${MODULE_FILE}")
  endforeach()

  set("WHEEL_FILES_${PYTHON_MODULE_WHEEL}"
      "${WHEEL_FILES_${PYTHON_MODULE_WHEEL}}"
      PARENT_SCOPE)

  add_custom_target(
    "${PYTHON_MODULE_TARGET_NAME}"
    DEPENDS ${PYTHON_MODULE_MODULE_FILES}
            ${PYTHON_MODULE_MODULE_GENERATED_FILES})
endfunction()

#
# Wheel generation
#
function(python_wheel)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs)
  cmake_parse_arguments(PYTHON_WHEEL "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

  set(WHEEL_OUTPUT_NAME
      "${PYTHON_WHEEL_NAME}-${PYTHON_WHL_VERSION}-py3-none-any.whl")
  configure_file(
    "pyproject_${PYTHON_WHEEL_NAME}.toml"
    "${CMAKE_BINARY_DIR}/build-wheel/${PYTHON_WHEEL_NAME}/pyproject.toml"
    COPYONLY)
  configure_file(
    "README_${PYTHON_WHEEL_NAME}.md"
    "${CMAKE_BINARY_DIR}/build-wheel/${PYTHON_WHEEL_NAME}/README.md" COPYONLY)

  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/${WHEEL_OUTPUT_NAME}"
    COMMAND python -m build --no-isolation --wheel --outdir
            "${CMAKE_BINARY_DIR}" &> /dev/null
    DEPENDS
      "${CMAKE_BINARY_DIR}/build-wheel/${PYTHON_WHEEL_NAME}/pyproject.toml"
      "${WHEEL_FILES_${PYTHON_WHEEL_NAME}}"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/build-wheel/${PYTHON_WHEEL_NAME}")

  add_custom_target("generate-${PYTHON_WHEEL_NAME}-wheel" ALL
                    DEPENDS "${CMAKE_BINARY_DIR}/${WHEEL_OUTPUT_NAME}")
  install(FILES "${CMAKE_BINARY_DIR}/${WHEEL_OUTPUT_NAME}"
          DESTINATION "${CMAKE_INSTALL_PREFIX}/share/python")
  install(
    CODE "execute_process(COMMAND \"${CMAKE_CURRENT_SOURCE_DIR}/wheel-install.sh\" \
\"${CMAKE_BINARY_DIR}/${WHEEL_OUTPUT_NAME}\")")
endfunction()

#
# Install scripts
#
configure_file("scripts/revng" "${CMAKE_BINARY_DIR}/bin/revng")

#
# Install revng
#
python_module(
  TARGET_NAME
  python-top-module
  WHEEL
  revng
  MODULE_INIT
  revng/__init__.py
  MODULE_FILES
  revng/py.typed)

#
# Install revng.internal
#
python_module(
  TARGET_NAME
  python-top-internal-module
  WHEEL
  revng_internal
  MODULE_INIT
  revng/internal/__init__.py
  MODULE_FILES
  revng/internal/py.typed)

#
# Install revng.model (including autogenerated classes)
#
set(PYTHON_MODEL_FILES
    revng/model/__init__.py revng/model/metaaddress.py revng/model/mixins.py
    revng/model/__init__.py revng/model/external.py)
python_module(
  TARGET_NAME
  python-model
  WHEEL
  revng
  MODULE_FILES
  ${PYTHON_MODEL_FILES}
  MODULE_GENERATED_FILES
  "${PYTHON_GENERATED_MODEL_PATH}")

#
# Install revng.model.migrations
#
set(PYTHON_MODEL_MIGRATIONS_FILES
    revng/model/migrations/__init__.py revng/model/migrations/versions/v2.py
    revng/model/migrations/versions/v3.py revng/model/migrations/versions/v4.py
    revng/model/migrations/versions/v5.py)
python_module(TARGET_NAME python-model-migrations WHEEL revng MODULE_FILES
              ${PYTHON_MODEL_MIGRATIONS_FILES})

#
# Install revng.tupletree
#
set(PYTHON_TUPLETREE_FILES revng/tupletree/__init__.py)
python_module(TARGET_NAME python-tupletree WHEEL revng MODULE_FILES
              ${PYTHON_TUPLETREE_FILES})

set(MERGE_DYNAMIC_MODULE_FILES
    revng/internal/cli/_commands/merge_dynamic/__init__.py
    revng/internal/cli/_commands/merge_dynamic/log.py
    revng/internal/cli/_commands/merge_dynamic/merge_dynamic.py
    revng/internal/cli/_commands/merge_dynamic/parsed_elf.py
    revng/internal/cli/_commands/merge_dynamic/util.py)
python_module(TARGET_NAME revng-merge-dynamic WHEEL revng_internal MODULE_FILES
              ${MERGE_DYNAMIC_MODULE_FILES})

#
# Install revng.project
#
set(REVNG_PROJECT_MODULE_FILES
    revng/project/__init__.py revng/project/cli_project.py
    revng/project/daemon_project.py revng/project/project.py
    revng/project/local_daemon_project.py)
python_module(TARGET_NAME revng-project WHEEL revng MODULE_FILES
              ${REVNG_PROJECT_MODULE_FILES})

#
# Install revng.ptml
#
set(REVNG_PTML_MODULE_FILES revng/ptml/__init__.py revng/ptml/actions.py
                            revng/ptml/parser.py revng/ptml/printer.py)
python_module(TARGET_NAME revng-ptml WHEEL revng MODULE_FILES
              ${REVNG_PTML_MODULE_FILES})

#
# Install revng.model_dump
#
set(DUMP_MODEL_MODULE_FILES
    revng/internal/cli/_commands/model_to_json/__init__.py
    revng/internal/cli/_commands/model_to_json/remap.py)
python_module(TARGET_NAME revng-dump-model WHEEL revng_internal MODULE_FILES
              ${DUMP_MODEL_MODULE_FILES})

#
# Install revng.cli
#
set(REVNG_CLI_MODULE_FILES
    revng/internal/cli/commands_registry.py revng/internal/cli/__init__.py
    revng/internal/cli/revng.py revng/internal/cli/support.py)
python_module(TARGET_NAME revng-python-cli WHEEL revng_internal MODULE_FILES
              ${REVNG_CLI_MODULE_FILES})

#
# Install revng.cli._commands
#
set(REVNG_CLI_COMMANDS_MODULE_FILES
    revng/internal/cli/_commands/translate.py
    revng/internal/cli/_commands/opt.py
    revng/internal/cli/_commands/pipeline_tools.py
    revng/internal/cli/_commands/override_by_name.py
    revng/internal/cli/_commands/daemon.py
    revng/internal/cli/_commands/llm_rename/__init__.py
    revng/internal/cli/_commands/llm_rename/prompt.tpl
    revng/internal/cli/_commands/hard_purge.py
    revng/internal/cli/_commands/process_docs_yaml/__init__.py
    revng/internal/cli/_commands/tar_to_yaml.py
    revng/internal/cli/_commands/test_docs/__init__.py
    revng/internal/cli/_commands/test_docs/doctest_runner.py
    revng/internal/cli/_commands/trace_run.py
    revng/internal/cli/_commands/model_compare.py
    revng/internal/cli/_commands/model_migrate.py
    revng/internal/cli/_commands/ptml.py)
python_module(TARGET_NAME revng-python-cli-commands WHEEL revng_internal
              MODULE_FILES ${REVNG_CLI_COMMANDS_MODULE_FILES})

set(REVNG_CLI_COMMANDS_IMPORT_IDB_MODULE_FILES
    revng/internal/cli/_commands/import_idb/__init__.py
    revng/internal/cli/_commands/import_idb/idb_converter.py)
python_module(
  TARGET_NAME revng-python-cli-commands-import-idb WHEEL revng_internal
  MODULE_FILES ${REVNG_CLI_COMMANDS_IMPORT_IDB_MODULE_FILES})

set(REVNG_CLI_COMMANDS_GENERATE_MIGRATION_MODULE_FILES
    revng/internal/cli/_commands/generate_migrations/__init__.py
    revng/internal/cli/_commands/generate_migrations/metaschema.yml
    revng/internal/cli/_commands/generate_migrations/migration.py.tpl)
python_module(
  TARGET_NAME revng-python-cli-commands-generate-migrations WHEEL
  revng_internal MODULE_FILES
  ${REVNG_CLI_COMMANDS_GENERATE_MIGRATION_MODULE_FILES})

set(REVNG_CLI_COMMANDS_FETCHDEBUGINFO_MODULE_FILES
    revng/internal/cli/_commands/fetch_debuginfo/pe.py
    revng/internal/cli/_commands/fetch_debuginfo/elf.py
    revng/internal/cli/_commands/fetch_debuginfo/common.py
    revng/internal/cli/_commands/fetch_debuginfo/__init__.py)
python_module(
  TARGET_NAME revng-python-cli-commands-fetchdebuginfo WHEEL revng_internal
  MODULE_FILES ${REVNG_CLI_COMMANDS_FETCHDEBUGINFO_MODULE_FILES})

set(REVNG_CLI_COMMANDS_GRAPHQL_MODULE_FILES
    revng/internal/cli/_commands/graphql/__init__.py
    revng/internal/cli/_commands/graphql/runner.py
    revng/internal/cli/_commands/graphql/daemon_handler.py)
python_module(
  TARGET_NAME revng-python-cli-commands-graphql WHEEL revng_internal
  MODULE_FILES ${REVNG_CLI_COMMANDS_GRAPHQL_MODULE_FILES})

set(REVNG_CLI_COMMANDS_MASS_TESTING_MODULE_FILES
    revng/internal/cli/_commands/mass_testing/__init__.py
    revng/internal/cli/_commands/mass_testing/generate_report/db.py
    revng/internal/cli/_commands/mass_testing/generate_report/meta.py
    revng/internal/cli/_commands/mass_testing/generate_report/stacktrace.py
    revng/internal/cli/_commands/mass_testing/generate_report/test_directory.py
    revng/internal/cli/_commands/mass_testing/generate_report/__init__.py
    revng/internal/cli/_commands/mass_testing/run.py
    revng/internal/cli/_commands/mass_testing/configure.py)
python_module(
  TARGET_NAME revng-python-cli-commands-mass-testing WHEEL revng_internal
  MODULE_FILES ${REVNG_CLI_COMMANDS_MASS_TESTING_MODULE_FILES})

#
# Install revng.api
#
set(REVNG_API_MODULE_FILES
    revng/internal/api/__init__.py
    revng/internal/api/_capi.py
    revng/internal/api/errors.py
    revng/internal/api/exceptions.py
    revng/internal/api/invalidations.py
    revng/internal/api/manager.py
    revng/internal/api/string_map.py
    revng/internal/api/syncing_manager.py
    revng/internal/api/synchronizer.py
    revng/internal/api/target.py
    revng/internal/api/utils.py)
python_module(TARGET_NAME revng-python-api WHEEL revng_internal MODULE_FILES
              ${REVNG_API_MODULE_FILES})

# Also copy PipelineC headers in the build directory so revng.api can use them
# from there
set(PIPELINE_C_HEADERS_SOURCE_DIR "${CMAKE_SOURCE_DIR}/include/revng/PipelineC")
set(PIPELINE_C_HEADERS_BUILD_DIR "${CMAKE_BINARY_DIR}/include/revng/PipelineC")
set(REQUIRED_PIPELINE_C_HEADERS ForwardDeclarationsC.h Prototypes.h)
file(MAKE_DIRECTORY "${PIPELINE_C_HEADERS_BUILD_DIR}")
foreach(HEADER ${REQUIRED_PIPELINE_C_HEADERS})
  configure_file("${PIPELINE_C_HEADERS_SOURCE_DIR}/${HEADER}"
                 "${PIPELINE_C_HEADERS_BUILD_DIR}/${HEADER}" COPYONLY)
endforeach()

#
# Install revng.daemon
#
set(REVNG_DAEMON_MODULE_FILES
    revng/internal/daemon/__init__.py revng/internal/daemon/multiqueue.py
    revng/internal/daemon/schema.graphql revng/internal/daemon/graphql.py
    revng/internal/daemon/util.py)
python_module(TARGET_NAME revng-python-daemon WHEEL revng_internal MODULE_FILES
              ${REVNG_DAEMON_MODULE_FILES})

#
# Install revng.internal.support
#
set(REVNG_INTERNAL_SUPPORT_MODULE_FILES
    revng/internal/support/__init__.py revng/internal/support/elf.py
    revng/internal/support/collect.py)
python_module(TARGET_NAME revng-python-internal-support WHEEL revng_internal
              MODULE_FILES ${REVNG_INTERNAL_SUPPORT_MODULE_FILES})

#
# Install revng.support
#
set(REVNG_SUPPORT_MODULE_FILES
    revng/support/__init__.py revng/support/artifacts.py
    revng/support/location.py)
python_module(TARGET_NAME revng-python-support WHEEL revng MODULE_FILES
              ${REVNG_SUPPORT_MODULE_FILES})

#
# Install revng.pipeline_description
#
set(PYTHON_PIPELINEDESCRIPTION_FILES revng/pipeline_description/__init__.py)
python_module(
  TARGET_NAME
  python-pipeline-description
  WHEEL
  revng
  MODULE_FILES
  ${PYTHON_PIPELINEDESCRIPTION_FILES}
  MODULE_GENERATED_FILES
  revng/pipeline_description/_generated.py)

#
# Generate actual wheel rules
#
python_wheel(NAME revng)
python_wheel(NAME revng_internal)
