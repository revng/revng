#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

cmake_minimum_required(VERSION 3.15.0)

project(revng-c)

#
# Install revng-c cmake files
#
install(FILES cmake/revngcConfig.cmake DESTINATION share/revngc/cmake)

#
# Export CMake targets
#
install(
  EXPORT revngc
  NAMESPACE revngc::
  DESTINATION share/revngc/cmake)

# Enable testing with ctest
enable_testing()

# This has to be first to get highest priority
include_directories(include/)

# Require revng
find_package(revng REQUIRED)

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

# Set build and install path
add_definitions("-DINSTALL_PATH=\"${CMAKE_INSTALL_PREFIX}\"")
add_definitions("-DBUILD_PATH=\"${CMAKE_BINARY_DIR}\"")

#
# Support files (share/revng)
#
make_directory("${CMAKE_BINARY_DIR}/share/revng/")
copy_to_build_and_install(
  FILES
  share/revng/pipelines
  share/revng/pipelines/revng-c-analyses.yml
  share/revng/pipelines/c-code-generation.yml
  share/revng/pipelines/decompiled-yaml-to-c.yml
  share/revng/pipelines/helpers-to-header.yml
  share/revng/pipelines/ir-canonicalization.yml
  share/revng/pipelines/model-to-header.yml
  share/revng/pipelines/opt-for-decompilation.yml
  share/revng/pipelines/promote-stack-pointer.yml
  share/revng/pipelines/remove-lifting-artifacts.yml
  share/revng/pipelines/segregate-stack-accesses.yml)

copy_to_build_and_install(PROGRAMS libexec/revng
                          "scripts/revng-check-decompiled-c")

#
# Install all of share/ as is
#
install(DIRECTORY share/ DESTINATION share/)

#
# Set compilation flags for the whole project
#

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")
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
# Include other cmake files
#

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

include(tests/Tests.cmake)
