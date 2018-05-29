#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_minimum_required(VERSION 2.8)

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
llvm_map_components_to_libnames(LLVM_LIBRARIES core support irreader ScalarOpts
  linker Analysis object transformutils)

# Build the support module for each architecture and in several configurations
set(CLANG "${LLVM_TOOLS_BINARY_DIR}/clang")

set(SUPPORT_MODULES_CONFIGS "normal;trace")
set(SUPPORT_MODULES_CONFIG_normal "")
set(SUPPORT_MODULES_CONFIG_trace "-DTRACE")

foreach(ARCH arm mips x86_64 i386)
  set(OUTPUT "early-linked-${ARCH}.ll")
  add_custom_command(OUTPUT "${OUTPUT}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/early-linked.c"
    COMMAND "${CLANG}"
    ARGS "${CMAKE_CURRENT_SOURCE_DIR}/early-linked.c"
         -o "${OUTPUT}"
         -S -emit-llvm -g
         -DTARGET_${ARCH}
         -I"${CMAKE_CURRENT_SOURCE_DIR}/")
  add_custom_target("early-linked-module-${OUTPUT}" ALL DEPENDS "${OUTPUT}")
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT}"
    DESTINATION share/revamb)

  foreach(CONFIG ${SUPPORT_MODULES_CONFIGS})
    set(OUTPUT "support-${ARCH}-${CONFIG}.ll")
    add_custom_command(OUTPUT "${OUTPUT}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/support.c"
      COMMAND "${CLANG}"
      ARGS "${CMAKE_CURRENT_SOURCE_DIR}/support.c"
           -o "${OUTPUT}"
           -S -emit-llvm -g
           -DTARGET_${ARCH}
           -I"${CMAKE_CURRENT_SOURCE_DIR}"
           ${SUPPORT_MODULES_CONFIG_${CONFIG}})
    add_custom_target("support-module-${OUTPUT}" ALL DEPENDS "${OUTPUT}")
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT}"
      DESTINATION share/revamb)
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror")

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
add_flag_if_available("-Wno-error=unused-variable")
add_flag_if_available("-Wno-error=unused-function")
add_flag_if_available("-Wno-error=return-type")
add_flag_if_available("-Wno-error=unused-but-set-variable")
add_flag_if_available("-Wno-error=maybe-uninitialized")
add_flag_if_available("-Wno-error=ignored-attributes")
add_flag_if_available("-Wno-delete-non-virtual-dtor")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
add_definitions("-D_FILE_OFFSET_BITS=64")
include_directories(argparse/)

CHECK_CXX_COMPILER_FLAG("-no-pie" COMPILER_SUPPORTS_NO_PIE)
if(COMPILER_SUPPORTS_NO_PIE)
  set(NO_PIE "-no-pie")
endif()

set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib${LLVM_LIBDIR_SUFFIX}")

add_executable(revamb ptcdump.cpp main.cpp debughelper.cpp variablemanager.cpp
  jumptargetmanager.cpp instructiontranslator.cpp codegenerator.cpp debug.cpp
  osra.cpp set.cpp simplifycomparisons.cpp reachingdefinitions.cpp
  functionboundariesdetection.cpp noreturnanalysis.cpp binaryfile.cpp
  generatedcodebasicinfo.cpp functioncallidentification.cpp argparse/argparse.c
  externaljumpshandler.cpp)
target_link_libraries(revamb dl m ${LLVM_LIBRARIES})
install(TARGETS revamb RUNTIME DESTINATION bin)

add_executable(revamb-dump dump.cpp collectcfg.cpp collectnoreturn.cpp debug.cpp
  collectfunctionboundaries.cpp debughelper.cpp stackanalysis.cpp
  generatedcodebasicinfo.cpp isolatefunctions.cpp argparse/argparse.c)
target_link_libraries(revamb-dump ${LLVM_LIBRARIES})
install(TARGETS revamb-dump RUNTIME DESTINATION bin)

configure_file(csv-to-ld-options "${CMAKE_BINARY_DIR}/csv-to-ld-options" COPYONLY)
configure_file(support.c "${CMAKE_BINARY_DIR}/support.c" COPYONLY)
configure_file(support.h "${CMAKE_BINARY_DIR}/support.h" COPYONLY)
configure_file(commonconstants.h "${CMAKE_BINARY_DIR}/commonconstants.h" COPYONLY)
configure_file(early-linked.c "${CMAKE_BINARY_DIR}/early-linked.c" COPYONLY)
configure_file(translate "${CMAKE_BINARY_DIR}/translate" COPYONLY)
configure_file(merge-dynamic.py "${CMAKE_BINARY_DIR}/merge-dynamic.py" COPYONLY)
install(PROGRAMS translate csv-to-ld-options merge-dynamic.py DESTINATION bin)
install(FILES support.c DESTINATION share/revamb)
install(FILES support.h DESTINATION share/revamb)
install(FILES commonconstants.h DESTINATION share/revamb)

# Remove -rdynamic
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)

include(tests/Tests.cmake)

add_subdirectory(docs/)
