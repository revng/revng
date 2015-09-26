cmake_minimum_required(VERSION 3.0)

project(revamb)

# Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)

# LLVM CMake stuff
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(LLVM_LIBRARIES core support)

# One day we won't need the source, but the required header files will be
# installed in the system.
set(QEMU_SOURCE_PATH "" CACHE PATH "Path to the QEMU source code.")

set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Werror -Wno-error=unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=return-type")
add_definitions("-D_FILE_OFFSET_BITS=64")

include_directories(argparse/
  "${QEMU_SOURCE_PATH}/tcg"
  "${QEMU_SOURCE_PATH}/linux-user")

add_executable(revamb ptcdump.cpp main.cpp ptctollvmir.cpp
  argparse/argparse.c)
target_link_libraries(revamb dl m ${LLVM_LIBRARIES})
