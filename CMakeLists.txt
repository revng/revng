#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_minimum_required(VERSION 2.8)

project(revng)

include(cmake/Common.cmake)
install(FILES cmake/revngConfig.cmake cmake/Common.cmake
  DESTINATION share/revng/cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# This has to be first to get highest priority
include_directories(include/)

# Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
  execute_process(COMMAND git ls-files
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
    configure_file(${CMAKE_SOURCE_DIR}/Doxyfile.in
      ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
    add_custom_target(doc
      ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen" VERBATIM)
  else()
    message(WARNING "Source directory is not a git repository, disabling Doxygen. Error was: ${GIT_LS_OUTPUT_STDERR}")
  endif()

endif(DOXYGEN_FOUND)

# LLVM CMake stuff
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
llvm_map_components_to_libnames(LLVM_LIBRARIES core support irreader ScalarOpts
  linker Analysis object transformutils)

# Build the support module for each architecture and in several configurations
set(CLANG "${LLVM_TOOLS_BINARY_DIR}/clang")

set(SUPPORT_MODULES_CONFIGS "normal;trace")
set(SUPPORT_MODULES_CONFIG_normal "")
set(SUPPORT_MODULES_CONFIG_trace "-DTRACE")

foreach(ARCH arm mips mipsel x86_64 i386 s390x)
  set(OUTPUT "early-linked-${ARCH}.ll")
  add_custom_command(OUTPUT "${OUTPUT}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/early-linked.c"
    COMMAND "${CLANG}"
    ARGS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/early-linked.c"
         -o "${OUTPUT}"
         -S -emit-llvm -g
         -DTARGET_${ARCH}
         -I"${CMAKE_CURRENT_SOURCE_DIR}/runtime"
         -I"${CMAKE_CURRENT_SOURCE_DIR}/include")
  add_custom_target("early-linked-module-${OUTPUT}" ALL DEPENDS "${OUTPUT}")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT}"
    DESTINATION share/revng)

  # Enable the support for C exceptions to avoid optimizations that break
  # exception support when linking a module with isolated functions
  foreach(CONFIG ${SUPPORT_MODULES_CONFIGS})
    set(OUTPUT "support-${ARCH}-${CONFIG}.ll")
    add_custom_command(OUTPUT "${OUTPUT}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/support.c"
      COMMAND "${CLANG}"
      ARGS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/support.c"
           -O2
           -fexceptions
           -o "${OUTPUT}"
           -S -emit-llvm -g
           -DTARGET_${ARCH}
           -I"${CMAKE_CURRENT_SOURCE_DIR}/runtime"
           -I"${CMAKE_CURRENT_SOURCE_DIR}/include"
           ${SUPPORT_MODULES_CONFIG_${CONFIG}})
    add_custom_target("support-module-${OUTPUT}" ALL DEPENDS "${OUTPUT}")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT}"
      DESTINATION share/revng)
  endforeach()
endforeach()

set(QEMU_INSTALL_PATH "/usr" CACHE PATH "Path to the QEMU installation.")
add_definitions("-DQEMU_INSTALL_PATH=\"${QEMU_INSTALL_PATH}\"")
add_definitions("-DINSTALL_PATH=\"${CMAKE_INSTALL_PREFIX}\"")
include_directories("${QEMU_INSTALL_PATH}/include/")

#
# Compiler options
#

# Basic compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Werror")

# Additional compiler options
include(CheckCXXCompilerFlag)
macro(add_flag_if_available flag)
  string(REPLACE "-" "_" NAME "${flag}")
  string(REPLACE "+" "_" NAME "${NAME}")
  string(REPLACE "=" "_" NAME "${NAME}")
  string(REPLACE "__" "_" NAME "${NAME}")
  string(TOUPPER "${NAME}" NAME)
  CHECK_CXX_COMPILER_FLAG("${flag}" IS_SUPPORTED_${NAME})
  if (IS_SUPPORTED_${NAME})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
  endif()
endmacro()

# Disable some warnings
add_flag_if_available("-Wno-maybe-uninitialized")
add_flag_if_available("-Wno-unused-local-typedefs")

# Add some extra warnings
add_flag_if_available("-Wstrict-aliasing")
add_flag_if_available("-fstrict-aliasing")

add_flag_if_available("-Wnon-virtual-dtor")
add_flag_if_available("-Wunreachable-code-break")
add_flag_if_available("-Winconsistent-missing-destructor-override")
add_flag_if_available("-Wnewline-eof")
add_flag_if_available("-Wmissing-prototypes")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
add_definitions("-D_FILE_OFFSET_BITS=64")

CHECK_CXX_COMPILER_FLAG("-no-pie" COMPILER_SUPPORTS_NO_PIE)
if(COMPILER_SUPPORTS_NO_PIE)
  set(NO_PIE "-no-pie")
endif()

include(CheckIncludeFiles)
CHECK_INCLUDE_FILES(valgrind/callgrind.h HAVE_VALGRIND_CALLGRIND_H)
if(HAVE_VALGRIND_CALLGRIND_H)
add_definitions("-DHAVE_VALGRIND_CALLGRIND_H")
endif()

set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib${LLVM_LIBDIR_SUFFIX}:\$ORIGIN/../lib${LLVM_LIBDIR_SUFFIX}/revng/analyses")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

configure_file(runtime/support.c "${CMAKE_BINARY_DIR}/support.c" COPYONLY)
configure_file(runtime/support.h "${CMAKE_BINARY_DIR}/support.h" COPYONLY)
configure_file(include/revng/Runtime/commonconstants.h "${CMAKE_BINARY_DIR}/commonconstants.h" COPYONLY)
configure_file(runtime/early-linked.c "${CMAKE_BINARY_DIR}/early-linked.c" COPYONLY)
configure_file(scripts/revng "${CMAKE_BINARY_DIR}/revng" COPYONLY)
configure_file(scripts/revng-merge-dynamic "${CMAKE_BINARY_DIR}/revng-merge-dynamic" COPYONLY)
install(PROGRAMS scripts/revng scripts/revng-merge-dynamic DESTINATION bin)
install(FILES runtime/support.c DESTINATION share/revng)
install(FILES runtime/support.h DESTINATION share/revng)
install(FILES include/revng/Runtime/commonconstants.h DESTINATION share/revng)

# Remove -rdynamic
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

include(tests/Tests.cmake)

install(EXPORT revng NAMESPACE revng:: DESTINATION share/revng/cmake)

add_subdirectory(docs/)
