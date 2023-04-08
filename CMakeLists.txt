#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_minimum_required(VERSION 3.15.0)

project(revng)

include(share/revng/cmake/Common.cmake)

#
# Compile flags
#

# These have to be first to get highest priority
include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_BINARY_DIR}/include")

add_definitions("-DINSTALL_PATH=\"${CMAKE_INSTALL_PREFIX}\"")

# Uncomment the following line if recursive coroutines make debugging hard
# add_definitions("-DDISABLE_RECURSIVE_COROUTINES")

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
add_flag_if_available("-Wimplicit-fallthrough")

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

#
# Link LLVM
#
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
link_directories(${LLVM_LIBRARY_DIRS})
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
  Passes
  TargetParser)

# Create share/revng/additional-bin-paths
set(ADDITIONAL_BIN_PATHS "${CMAKE_BINARY_DIR}/share/revng/additional-bin-paths")
file(WRITE "${ADDITIONAL_BIN_PATHS}" "libexec/revng\n")

file(RELATIVE_PATH RELATIVE_LLVM_TOOLS_BINARY_DIR "${CMAKE_INSTALL_PREFIX}"
     "${LLVM_TOOLS_BINARY_DIR}")
file(APPEND "${ADDITIONAL_BIN_PATHS}" "${RELATIVE_LLVM_TOOLS_BINARY_DIR}\n")

# Create additional-search-prefixes
file(WRITE "${CMAKE_BINARY_DIR}/additional-search-prefixes"
     "${CMAKE_INSTALL_PREFIX}\n")

file(RELATIVE_PATH RELATIVE_LLVM_TOOLS_BINARY_DIR "${CMAKE_INSTALL_PREFIX}"
     "${LLVM_TOOLS_BINARY_DIR}")
file(APPEND "${ADDITIONAL_BIN_PATHS}" "${RELATIVE_LLVM_TOOLS_BINARY_DIR}\n")

#
# share/revng
#
add_custom_target(copy_share ALL COMMAND cp -Tar "${CMAKE_SOURCE_DIR}/share/"
                                         "${CMAKE_BINARY_DIR}/share/")

install(
  DIRECTORY "${CMAKE_BINARY_DIR}/share/"
  DESTINATION share/
  USE_SOURCE_PERMISSIONS)

# Export CMake targets
install(
  EXPORT revng
  NAMESPACE revng::
  DESTINATION share/revng/cmake)

# Build the support module for each architecture and in several configurations
set(CLANG "${LLVM_TOOLS_BINARY_DIR}/clang")

set(SUPPORT_MODULES_CONFIGS "normal;trace")
set(SUPPORT_MODULES_CONFIG_normal "")
set(SUPPORT_MODULES_CONFIG_trace "-DTRACE")

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
    DEPENDS "${CMAKE_SOURCE_DIR}/share/revng/early-linked.c"
    COMMAND
      "${CLANG}" ARGS "${CMAKE_SOURCE_DIR}/share/revng/early-linked.c" -o
      "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}" -S -emit-llvm -g
      -DTARGET_${ARCH} -I"${CMAKE_SOURCE_DIR}/share/revng"
      -I"${CMAKE_CURRENT_SOURCE_DIR}/include")
  add_custom_target("early-linked-module-${OUTPUT}" ALL
                    DEPENDS "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}")
  add_dependencies(revng-all-binaries "early-linked-module-${OUTPUT}")

  # Enable the support for C exceptions to avoid optimizations that break
  # exception support when linking a module with isolated functions
  foreach(CONFIG ${SUPPORT_MODULES_CONFIGS})
    set(OUTPUT "support-${ARCH}-${CONFIG}.ll")
    add_custom_command(
      OUTPUT "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}"
      DEPENDS "${CMAKE_SOURCE_DIR}/share/revng/support.c"
      COMMAND
        "${CLANG}" ARGS "${CMAKE_SOURCE_DIR}/share/revng/support.c" -O2
        -fexceptions -o "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}" -S
        -emit-llvm -g -DTARGET_${ARCH} -I"${CMAKE_SOURCE_DIR}/share/revng"
        -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${SUPPORT_MODULES_CONFIG_${CONFIG}})
    add_custom_target("support-module-${OUTPUT}" ALL
                      DEPENDS "${CMAKE_BINARY_DIR}/share/revng/${OUTPUT}")
    add_dependencies(revng-all-binaries "support-module-${OUTPUT}")
  endforeach()
endforeach()

# Custom command to create .clang-format file from revng-check-conventions
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/share/revng/.clang-format"
  DEPENDS "${CMAKE_SOURCE_DIR}/libexec/revng/revng-check-conventions"
          "${CMAKE_SOURCE_DIR}/share/revng/clang-format-style-file" copy_share
          copy_libexec
  COMMAND
    "${CMAKE_BINARY_DIR}/libexec/revng/revng-check-conventions" ARGS
    --print-clang-format-config >
    "${CMAKE_BINARY_DIR}/share/revng/.clang-format")

add_custom_target(clang-format-dot-file ALL
                  DEPENDS "${CMAKE_BINARY_DIR}/share/revng/.clang-format")

#
# libexec/revng
#
add_custom_target(
  copy_libexec ALL COMMAND cp -Tar "${CMAKE_SOURCE_DIR}/libexec/"
                           "${CMAKE_BINARY_DIR}/libexec/")
install(
  DIRECTORY "${CMAKE_BINARY_DIR}/libexec/"
  DESTINATION libexec/
  USE_SOURCE_PERMISSIONS)

#
# Export information useful for subdirectories
#
set(MODEL_SCHEMA_PATH "${CMAKE_BINARY_DIR}/model-schema.yml")
set(MODEL_JSONSCHEMA_PATH "${CMAKE_BINARY_DIR}/model-jsonschema.yml")
set(PYTHON_GENERATED_MODEL_PATH revng/model/v1/_generated.py)

#
# Enable CTest
#
enable_testing()

find_package(Python REQUIRED)
file(RELATIVE_PATH PYTHON_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}"
     "${Python_SITELIB}")

#
# Proceed to subdirectories
#
add_subdirectory(docs)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)
add_subdirectory(scripts)
add_subdirectory(tests)
add_subdirectory(tools)
add_subdirectory(typescript)
