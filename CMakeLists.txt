#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_minimum_required(VERSION 3.15.0)

project(revng)

include(cmake/Common.cmake)
install(FILES cmake/revngConfig.cmake cmake/Common.cmake
              cmake/TupleTreeGenerator.cmake DESTINATION share/revng/cmake)

# This has to be first to get highest priority
include_directories(include/)
include_directories("${CMAKE_BINARY_DIR}/include")

# Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  execute_process(
    COMMAND git ls-files
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE GIT_LS_EXIT_CODE
    OUTPUT_VARIABLE GIT_LS_OUTPUT
    ERROR_VARIABLE GIT_LS_OUTPUT_STDERR)

  if(GIT_LS_EXIT_CODE EQUAL "0")
    string(REGEX REPLACE "\n" ";" GIT_LS_OUTPUT "${GIT_LS_OUTPUT}")
    set(DOXYGEN_INPUTS "")
    foreach(FILE ${GIT_LS_OUTPUT})
      set(DOXYGEN_INPUTS "${DOXYGEN_INPUTS} ${CMAKE_SOURCE_DIR}/${FILE}")
    endforeach(FILE)
    configure_file(${CMAKE_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile
                   @ONLY)
    add_custom_target(
      doc
      ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
      VERBATIM)
  else()
    message(
      WARNING
        "Source directory is not a git repository, disabling Doxygen. Error was: ${GIT_LS_OUTPUT_STDERR}"
    )
  endif()

endif(DOXYGEN_FOUND)

# LLVM CMake stuff
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(
  LLVM_LIBRARIES
  core
  support
  irreader
  ScalarOpts
  linker
  Analysis
  object
  transformutils
  BitWriter
  InstCombine
  CodeGen
  Passes)

# Build the support module for each architecture and in several configurations
set(CLANG "${LLVM_TOOLS_BINARY_DIR}/clang")

set(SUPPORT_MODULES_CONFIGS "normal;trace")
set(SUPPORT_MODULES_CONFIG_normal "")
set(SUPPORT_MODULES_CONFIG_trace "-DTRACE")

make_directory("${CMAKE_BINARY_DIR}/share/revng/")

foreach(
  ARCH
  aarch64
  arm
  mips
  mipsel
  x86_64
  i386
  s390x)
  set(OUTPUT "early-linked-${ARCH}.ll")
  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/early-linked.c"
    COMMAND
      "${CLANG}" ARGS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/early-linked.c" -o
      "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}" -S -emit-llvm -g
      -DTARGET_${ARCH} -I"${CMAKE_CURRENT_SOURCE_DIR}/runtime"
      -I"${CMAKE_CURRENT_SOURCE_DIR}/include")
  add_custom_target("early-linked-module-${OUTPUT}" ALL
                    DEPENDS "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}")
  add_dependencies(revng-all-binaries "early-linked-module-${OUTPUT}")
  install(FILES "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}"
          DESTINATION share/revng)

  # Enable the support for C exceptions to avoid optimizations that break
  # exception support when linking a module with isolated functions
  foreach(CONFIG ${SUPPORT_MODULES_CONFIGS})
    set(OUTPUT "support-${ARCH}-${CONFIG}.ll")
    add_custom_command(
      OUTPUT "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/support.c"
      COMMAND
        "${CLANG}" ARGS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/support.c" -O2
        -fexceptions -o "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}" -S
        -emit-llvm -g -DTARGET_${ARCH} -I"${CMAKE_CURRENT_SOURCE_DIR}/runtime"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${SUPPORT_MODULES_CONFIG_${CONFIG}})
    add_custom_target("support-module-${OUTPUT}" ALL
                      DEPENDS "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}")
    add_dependencies(revng-all-binaries "support-module-${OUTPUT}")
    install(FILES "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}"
            DESTINATION share/revng)
  endforeach()
endforeach()

add_definitions("-DINSTALL_PATH=\"${CMAKE_INSTALL_PREFIX}\"")

# Uncomment the following line if recursive coroutines make debugging hard
# add_definitions("-DDISABLE_RECURSIVE_COROUTINES")

#
# Compiler options
#

# Remove -rdynamic
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

# Basic compiler options
# cmake-format: off
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
# cmake-format: on
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_flag_if_available("-Wno-unused-local-typedefs")
endif()

# Disable some warnings
add_flag_if_available("-Wno-unused-parameter")
add_flag_if_available("-Wno-unused-variable")
add_flag_if_available("-Wno-maybe-uninitialized")
add_flag_if_available("-Wno-init-list-lifetime")
add_flag_if_available("-Wno-ambiguous-reversed-operator")

# Add some extra warnings
add_flag_if_available("-Wstrict-aliasing")
add_flag_if_available("-fstrict-aliasing")

add_flag_if_available("-Wnon-virtual-dtor")
add_flag_if_available("-Wunreachable-code-break")
add_flag_if_available("-Winconsistent-missing-destructor-override")
add_flag_if_available("-Wnewline-eof")
add_flag_if_available("-Wmissing-prototypes")

add_definitions("-D_FILE_OFFSET_BITS=64")

check_cxx_compiler_flag("-no-pie" COMPILER_SUPPORTS_NO_PIE)
if(COMPILER_SUPPORTS_NO_PIE)
  set(NO_PIE "-no-pie")
endif()

include(CheckIncludeFiles)
check_include_files(valgrind/callgrind.h HAVE_VALGRIND_CALLGRIND_H)
if(HAVE_VALGRIND_CALLGRIND_H)
  add_definitions("-DHAVE_VALGRIND_CALLGRIND_H")
endif()

set(VERSION 0.0.0)

#
# Support files (share/revng)
#
copy_to_build_and_install(
  FILES
  share/revng
  runtime/support.c
  runtime/support.h
  scripts/clang-format-style-file
  scripts/flake8-config
  scripts/isort.cfg)

copy_to_build_and_install(
  FILES
  share/revng/pipelines
  share/revng/pipelines/isolate-translate.yml
  share/revng/pipelines/translate.yml
  share/revng/pipelines/enforce-abi.yml
  share/revng/pipelines/yield-assembly.yml
  share/revng/pipelines/yield-cfg.yml)

configure_file(runtime/early-linked.c
               "${CMAKE_BINARY_DIR}/share/revng/early-linked.c" COPYONLY)

# Custom command to create .clang-format file from revng-check-conventions
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/share/revng/.clang-format"
  DEPENDS "${CMAKE_BINARY_DIR}/libexec/revng/revng-check-conventions"
          "${CMAKE_BINARY_DIR}/share/revng/clang-format-style-file"
  COMMAND
    "${CMAKE_BINARY_DIR}/libexec/revng/revng-check-conventions" ARGS
    --print-clang-format-config >
    "${CMAKE_BINARY_DIR}/share/revng/.clang-format")

add_custom_target(clang-format-dot-file ALL
                  DEPENDS "${CMAKE_BINARY_DIR}/share/revng/.clang-format")
install(FILES "${CMAKE_BINARY_DIR}/share/revng/.clang-format"
        DESTINATION share/revng)

#
# Executable scripts
#
copy_to_build_and_install(PROGRAMS libexec/revng
                          "scripts/revng-check-conventions")

#
# Export CMake targets
#
install(
  EXPORT revng
  NAMESPACE revng::
  DESTINATION share/revng/cmake)

#
# Export information useful for subdirectories
#
set(MODEL_SCHEMA_PATH "${CMAKE_BINARY_DIR}/model-schema.yml")
set(MODEL_JSONSCHEMA_PATH "${CMAKE_BINARY_DIR}/model-jsonschema.yml")
set(PYTHON_GENERATED_MODEL_PATH revng/model/v1/_generated.py)

#
# Include other CMake files
#
add_subdirectory(scripts)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(docs/)
add_subdirectory(python)
add_subdirectory(typescript)

include(${CMAKE_INSTALL_PREFIX}/share/revng/qa/cmake/revng-qa.cmake)

include(tests/Tests.cmake)
