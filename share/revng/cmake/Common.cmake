#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

add_custom_target(revng-all-binaries)

macro(prepend_target_property TARGET PROPERTY VALUE SEPARATOR)
  get_target_property(TMP "${TARGET}" "${PROPERTY}")
  if("${TMP}" STREQUAL "TMP-NOTFOUND")
    set(TMP "")
  endif()
  if(NOT "${TMP}" STREQUAL "")
    set(TMP "${SEPARATOR}${TMP}")
  endif()
  set_target_properties("${TARGET}" PROPERTIES "${PROPERTY}" "${VALUE}${TMP}")
endmacro()

macro(append_target_property TARGET PROPERTY VALUE SEPARATOR)
  get_target_property(TMP "${TARGET}" "${PROPERTY}")
  if("${TMP}" STREQUAL "TMP-NOTFOUND")
    set(TMP "")
  endif()
  if(NOT "${TMP}" STREQUAL "")
    set(TMP "${TMP}${SEPARATOR}")
  endif()
  set_target_properties("${TARGET}" PROPERTIES "${PROPERTY}" "${TMP}${VALUE}")
endmacro()

macro(revng_add_library NAME TYPE EXPORT_NAME)

  add_library("${NAME}" "${TYPE}" ${ARGN})
  add_dependencies(revng-all-binaries "${NAME}")
  target_include_directories("${NAME}" INTERFACE $<INSTALL_INTERFACE:include/>)
  prepend_target_property("${NAME}" BUILD_RPATH
                          "\$ORIGIN:\$ORIGIN/revng/analyses" ":")
  if(NOT "${CMAKE_INSTALL_RPATH}" STREQUAL "")
    append_target_property("${NAME}" BUILD_RPATH "${CMAKE_INSTALL_RPATH}" ":")
  endif()

  set_target_properties("${NAME}" PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                             "${CMAKE_BINARY_DIR}/lib")

  install(
    TARGETS "${NAME}"
    EXPORT "${EXPORT_NAME}"
    LIBRARY DESTINATION lib/
    ARCHIVE DESTINATION lib/)

endmacro()

# Helper macro to create a new library containing analyses to be employed in
# revng-opt
macro(revng_add_analyses_library NAME EXPORT_NAME)

  add_library("${NAME}" SHARED ${ARGN})
  add_dependencies(revng-all-binaries "${NAME}")
  target_include_directories("${NAME}" INTERFACE $<INSTALL_INTERFACE:include/>)
  prepend_target_property("${NAME}" BUILD_RPATH "\$ORIGIN/../../:\$ORIGIN" ":")
  if(NOT "${CMAKE_INSTALL_RPATH}" STREQUAL "")
    append_target_property("${NAME}" BUILD_RPATH "${CMAKE_INSTALL_RPATH}" ":")
  endif()

  set_target_properties(
    "${NAME}" PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                         "${CMAKE_BINARY_DIR}/lib/revng/analyses")

  install(
    TARGETS "${NAME}"
    EXPORT "${EXPORT_NAME}"
    LIBRARY DESTINATION lib/revng/analyses)

endmacro()

macro(revng_add_executable_internal NAME TARGET_PATH)
  # Compute how many ../ to get to root
  set(RELATIVE_TO_ROOT "")
  if(NOT "${TARGET_PATH}" STREQUAL "")
    # Normalize TARGET_PATH
    string(REGEX REPLACE "^/+" "" TARGET_PATH "${TARGET_PATH}")
    string(REGEX REPLACE "/+$" "" TARGET_PATH "${TARGET_PATH}")

    # Count slashes
    string(REPLACE "/" "" TARGET_PATH_WITHOUT_SLASHES "${TARGET_PATH}")
    string(LENGTH "${TARGET_PATH}" TARGET_PATH_LENGTH)
    string(LENGTH "${TARGET_PATH_WITHOUT_SLASHES}"
                  TARGET_PATH_WITHOUT_SLASHES_LENGTH)
    math(EXPR DEPTH
         "${TARGET_PATH_LENGTH} - ${TARGET_PATH_WITHOUT_SLASHES_LENGTH}")

    foreach(IGNORE RANGE "${DEPTH}")
      set(RELATIVE_TO_ROOT "${RELATIVE_TO_ROOT}../")
    endforeach()
  endif()

  add_executable("${NAME}" ${ARGN})
  append_target_property("${NAME}" "LINK_FLAGS" "-pie" " ")

  add_dependencies(revng-all-binaries "${NAME}")

  # Set BUILD_RPATH
  prepend_target_property(
    "${NAME}"
    BUILD_RPATH
    "\$ORIGIN/${RELATIVE_TO_ROOT}lib/:\$ORIGIN/${RELATIVE_TO_ROOT}lib/revng/analyses/"
    ":")
  if(NOT "${CMAKE_INSTALL_RPATH}" STREQUAL "")
    append_target_property("${NAME}" BUILD_RPATH "${CMAKE_INSTALL_RPATH}" ":")
  endif()

  # Build in the desired directory
  set_target_properties(
    "${NAME}" PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                         "${CMAKE_BINARY_DIR}/${TARGET_PATH}")

endmacro()

macro(revng_add_executable NAME)
  set(TARGET_PATH libexec/revng)
  revng_add_executable_internal("${NAME}" "${TARGET_PATH}" ${ARGN})
  install(TARGETS "${NAME}" RUNTIME DESTINATION "${TARGET_PATH}")
endmacro()

macro(revng_add_test_executable NAME)
  revng_add_executable_internal("${NAME}" "" ${ARGN})
endmacro()

# This macro installs all the files matching the pattern in the extra arguments
macro(install_pattern)

  file(
    GLOB_RECURSE FILES_TO_INSTALL
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    ${ARGN})

  file(RELATIVE_PATH RELATIVE_SOURCE_DIR ${CMAKE_SOURCE_DIR}
       ${CMAKE_CURRENT_SOURCE_DIR})

  foreach(FILE ${FILES_TO_INSTALL})
    get_filename_component(INSTALL_PATH "${FILE}" DIRECTORY)
    install(FILES "${FILE}" DESTINATION ${RELATIVE_SOURCE_DIR}/${INSTALL_PATH})
  endforeach(FILE)

endmacro(install_pattern)

# Additional compiler options
include(CheckCXXCompilerFlag)
macro(add_flag_if_available flag)
  string(REPLACE "-" "_" NAME "${flag}")
  string(REPLACE "+" "_" NAME "${NAME}")
  string(REPLACE "=" "_" NAME "${NAME}")
  string(REPLACE "__" "_" NAME "${NAME}")
  string(TOUPPER "${NAME}" NAME)
  check_cxx_compiler_flag("${flag}" IS_SUPPORTED_${NAME})
  if(IS_SUPPORTED_${NAME})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
  endif()
endmacro()

include("${CMAKE_CURRENT_LIST_DIR}/TupleTreeGenerator.cmake")

function(copy_to_build_and_install INSTALL_TYPE DESTINATION)
  foreach(INPUT_FILE ${ARGN})
    make_directory("${CMAKE_BINARY_DIR}/${DESTINATION}")
    configure_file("${INPUT_FILE}" "${CMAKE_BINARY_DIR}/${DESTINATION}"
                   COPYONLY)
    install("${INSTALL_TYPE}" "${INPUT_FILE}" DESTINATION "${DESTINATION}")
  endforeach()
endfunction()

macro(revng_add_test)
  set(options OPTIONAL FAST)
  set(oneValueArgs DESTINATION RENAME)
  set(multiValueArgs TARGETS CONFIGURATIONS)
  cmake_parse_arguments(REVNG_TEST "" "NAME" "" ${ARGN})
  add_test(${ARGN})
  set_tests_properties(
    "${REVNG_TEST_NAME}"
    PROPERTIES ENVIRONMENT
               "REVNG_OPTIONS=--debug-log=verify $ENV{REVNG_OPTIONS}")
endmacro()
