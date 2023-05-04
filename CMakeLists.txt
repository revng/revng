#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Remove -rdynamic
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

# Uncomment the following line if recursive coroutines make debugging hard
# add_definitions("-DDISABLE_RECURSIVE_COROUTINES")

# Basic compiler options
# cmake-format: off
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
# cmake-format: on
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra ")

# Enable some additional warnings
add_flag_if_available("-Wimplicit-fallthrough")
add_flag_if_available("-Wnon-virtual-dtor")
add_flag_if_available("-Winconsistent-missing-destructor-override")
add_flag_if_available("-Wnewline-eof")
add_flag_if_available("-Wmissing-prototypes")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstrict-aliasing ")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

#
# Link LLVM
#

# LLVM and clang CMake stuff
find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
llvm_map_components_to_libnames(
  LLVM_LIBRARIES
  Analysis
  CodeGen
  InstCombine
  ScalarOpts
  core
  irreader
  linker
  object
  support
  transformutils)

# MLIR CMake stuff
find_package(MLIR REQUIRED CONFIG)
include_directories(${MLIR_INCLUDE_DIRS})

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
