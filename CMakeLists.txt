#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_minimum_required(VERSION 3.15.0)

project(revng-c)

# Require revng
find_package(revng REQUIRED)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG ON)

if(${TUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG})
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -DTUPLE_TREE_GENERATOR_EMIT_TRACKING_DEBUG")
endif()

# Uncomment the following line if errors like `Couldn't find method
# SomeType::method` make debugging hard
#
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-limit-debug-info")

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

# MLIR CMake stuff
find_package(MLIR REQUIRED CONFIG)
include_directories(${MLIR_INCLUDE_DIRS})

#
# Link Clang
#
find_package(Clang REQUIRED CONFIG)

#
# Component hash
#
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/share/revng/component-hashes")
set(COMPONENT_HASH_PATH
    "${CMAKE_BINARY_DIR}/share/revng/component-hashes/revng-c")
add_custom_command(
  OUTPUT "${COMPONENT_HASH_PATH}"
  COMMAND
    sh -c
    ' (git -C "${CMAKE_SOURCE_DIR}" rev-parse HEAD || echo "\"${CMAKE_PROJECT_VERSION}\"") '
    > "${COMPONENT_HASH_PATH}")
add_custom_target(generate-component-hash ALL DEPENDS "${COMPONENT_HASH_PATH}")
install(FILES "${COMPONENT_HASH_PATH}"
        DESTINATION "${CMAKE_INSTALL_DIR}/share/revng/component-hashes/")

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
  EXPORT revngc
  NAMESPACE revngc::
  DESTINATION share/revngc/cmake)

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
# Enable CTest
#
enable_testing()

#
# Proceed to subdirectories
#
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(tools)
