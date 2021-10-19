#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Declares a python module as a target
# Named arguments:
#  - TARGET_NAME: the name of the CMake target to be created
#  - MODULE_FILES: list of relative paths of the python module sources
#  - MODULE_GENERATED_FILES: list of relative paths of the python module generated sources
# The module sources will be copied to the build directory (in the python folder)
# and installed in /lib/python.
function(python_module)
  set(options)
  set(oneValueArgs TARGET_NAME)
  set(multiValueArgs MODULE_FILES MODULE_GENERATED_FILES)
  cmake_parse_arguments(PYTHON_MODULE "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  foreach(MODULE_FILE ${PYTHON_MODULE_MODULE_FILES})
    configure_file("${MODULE_FILE}" "${CMAKE_BINARY_DIR}/python/${MODULE_FILE}" COPYONLY)
  endforeach()

  foreach(MODULE_FILE ${PYTHON_MODULE_MODULE_FILES} ${PYTHON_MODULE_MODULE_GENERATED_FILES})
    # Copy module in the root at install time
    get_filename_component(MODULE_FILE_DIR "${MODULE_FILE}" DIRECTORY)
    install(
      FILES "${CMAKE_BINARY_DIR}/python/${MODULE_FILE}"
      DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python/${MODULE_FILE_DIR}"
    )
  endforeach()
  add_custom_target("${PYTHON_MODULE_TARGET_NAME}" DEPENDS ${PYTHON_MODULE_MODULE_FILES} ${PYTHON_MODULE_MODULE_GENERATED_FILES})
endfunction()

# -- Install revng-merge-dynamic
set(MERGE_DYNAMIC_MODULE_FILES
  revng/merge_dynamic/__init__.py
  revng/merge_dynamic/__main__.py
  revng/merge_dynamic/log.py
  revng/merge_dynamic/merge_dynamic.py
  revng/merge_dynamic/parsed_elf.py
  revng/merge_dynamic/util.py
)
python_module(
  TARGET_NAME revng-merge-dynamic
  MODULE_FILES ${MERGE_DYNAMIC_MODULE_FILES}
)
add_dependencies(revng-lift revng-merge-dynamic)

# -- Install revng-dump-model
set(DUMP_MODEL_MODULE_FILES
  "revng/revng_dump_model/__init__.py"
  "revng/revng_dump_model/__main__.py"
  "revng/revng_dump_model/dump_model.py"
)
python_module(
  TARGET_NAME revng-dump-model
  MODULE_FILES ${DUMP_MODEL_MODULE_FILES}
)
add_dependencies(revng-lift revng-dump-model)
