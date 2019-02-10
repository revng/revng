cmake_minimum_required(VERSION 2.8)

find_package(revng REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

install(FILES cmake/revngcConfig.cmake DESTINATION share/revngc/cmake)
install(EXPORT revngc NAMESPACE revngc:: DESTINATION share/revngc/cmake)

include_directories(include/)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

# LLVM CMake stuff
llvm_map_components_to_libnames(LLVM_LIBRARIES core support irreader ScalarOpts
  linker Analysis object transformutils)

add_subdirectory(include)
add_subdirectory(lib)
