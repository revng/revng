cmake_minimum_required(VERSION 2.8)

find_package(LLVM REQUIRED CONFIG)

set(CMAKE_PREFIX_PATH "${CMAKE_INSTALL_PATH}/share/clang/cmake/" ${CMAKE_INSTALL_PATH})
find_package(Clang REQUIRED)

find_library(REVNG_SUPPORT_LIBRARY revngSupport)
find_library(REVNG_DEBUGHELPER_LIBRARY revngDebugHelper)

include_directories(include/)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

# LLVM CMake stuff
llvm_map_components_to_libnames(LLVM_LIBRARIES core support irreader ScalarOpts
  linker Analysis object transformutils)

add_subdirectory(lib)
