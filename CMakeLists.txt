cmake_minimum_required(VERSION 3.15.0)

project(revng-c)

enable_testing()

find_package(revng REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra ")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstrict-aliasing ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wstrict-aliasing ")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")

install(FILES cmake/revngcConfig.cmake DESTINATION share/revngc/cmake)
install(EXPORT revngc NAMESPACE revngc:: DESTINATION share/revngc/cmake)

install(FILES share/revngc/revng-c-include.c DESTINATION share/revngc/)
configure_file(share/revngc/revng-c-include.c
  ${CMAKE_BINARY_DIR}/share/revngc/revng-c-include.c
  COPYONLY)

add_definitions("-DINSTALL_PATH=\"${CMAKE_INSTALL_PREFIX}\"")
add_definitions("-DBUILD_PATH=\"${CMAKE_BINARY_DIR}\"")

include_directories(include/)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

# LLVM CMake stuff
llvm_map_components_to_libnames(LLVM_LIBRARIES core support irreader ScalarOpts
  linker Analysis object transformutils)

add_subdirectory(include)
add_subdirectory(lib)

include(tests/Tests.cmake)
