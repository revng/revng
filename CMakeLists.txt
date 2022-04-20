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
find_package(Clang REQUIRED CONFIG)
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

# ##############################################################################
# Handle assets for recompilation #
# ##############################################################################
# TODO: For now we only put these assets in the build directory for tests. If we
# ever want to allow recompilation in production we need to decide how to
# provide these assets to users too.
set(ASSETS_DIR assets)
# Copy the recompilation headers in the build directory
file(COPY ${CMAKE_SOURCE_DIR}/${ASSETS_DIR}/recompilation-headers
     DESTINATION ${CMAKE_BINARY_DIR}/${ASSETS_DIR})
# Configure the compile_flags blueprint
configure_file(${CMAKE_SOURCE_DIR}/${ASSETS_DIR}/compile_flags.txt.in
               ${CMAKE_BINARY_DIR}/${ASSETS_DIR}/compile_flags.txt.in)
# Strip leading comments, used for documentation, away from the compile flags
# blueprint, actually producing the real compile_flags.txt 1. Parse the file,
# filling a list of lines.
file(STRINGS ${CMAKE_BINARY_DIR}/${ASSETS_DIR}/compile_flags.txt.in
     RECOMPILATION_COMPILE_FLAGS)
# A. Drop lines starting with '#' and lines with only whitespace
list(FILTER RECOMPILATION_COMPILE_FLAGS EXCLUDE REGEX "^#.*")
list(FILTER RECOMPILATION_COMPILE_FLAGS EXCLUDE REGEX "^[ \t\r\n]*$")
# B. Splat the remaining lines
list(JOIN RECOMPILATION_COMPILE_FLAGS "\n" RECOMPILATION_COMPILE_FLAGS)
# C. Generate the acutal compile_flags.txt
file(WRITE ${CMAKE_BINARY_DIR}/${ASSETS_DIR}/compile_flags.txt
     ${RECOMPILATION_COMPILE_FLAGS})

#
# Set compilation flags for the whole project
#

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra ")

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

include(${CMAKE_INSTALL_PREFIX}/share/revng/qa/cmake/revng-qa.cmake)

#
# Define tests with revng-qa artifacts
#

# Macro for creating the tests for decompilation pipeline
macro(add_decompilation_pipeline_test CATEGORY INPUT_FILE CONFIGURATION OUTPUT
      TARGET_NAME)

  # setup a directory for all the output files for this family of tests
  get_filename_component(FILENAME ${INPUT_FILE} NAME_WLE)
  set(OUTFOLDER ${CATEGORY}/${FILENAME}/${CONFIGURATION})
  make_directory(${OUTFOLDER})

  # prepare enforced .bc for decompilation
  add_test(
    NAME prepare_${TARGET_NAME}
    COMMAND
      revng --prefix=${CMAKE_BINARY_DIR} opt
      # Remove revng's artifacts from IR
      -filter-for-decompilation -remove-newpc-calls -remove-exception-calls
      -remove-cpu-loop-store -remove-llvm-dbg-intrinsics -make-env-null
      # Stack promotion
      -measure-stack-size-at-call-sites -promote-stack-pointer
      # Optimize IR with LLVM's passes
      -remove-extractvalues -dce -simplifycfg -dse -instcombine -sroa
      --sroa-aggressive-phis-selects -instsimplify -jump-threading -licm
      -unreachableblockelim -instcombine -early-cse -simplifycfg
      # apply type shrinking
      -type-shrinking --min-width 8 -early-cse -instsimplify -gvn -instsimplify
      -dse -dce
      # Remove LLVM's artifacts from IR
      -remove-llvmassume-calls -dce
      # Stack analysis and segregation
      -instrument-stack-accesses -instcombine -compute-stack-accesses-bounds
      -detect-stack-size -segregate-stack-accesses -cleanup-stack-size-markers
      -dce -sroa -instcombine -sroa -simplifycfg -restore-extractvalues
      -strip-dead-prototypes ${INPUT_FILE} -o ${OUTFOLDER}/prepared.bc)
  set_property(
    TEST prepare_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP PrepareDecompilation_fixture_${TARGET_NAME})

  if("${CATEGORY}" STREQUAL "tests_analysis_PromoteStackPointer"
     AND "${CONFIGURATION}" STREQUAL "dynamic_native")
    get_filename_component(BASENAME "${INPUT_FILE}" NAME_WE)
    add_test(
      NAME psp_${TARGET_NAME}
      COMMAND
        sh -c
        "revng opt -S ${OUTFOLDER}/prepared.bc | revng model dump | revng model compare - ${CMAKE_SOURCE_DIR}/tests/Analysis/PromoteStackPointer/${BASENAME}.yml"
    )
    set_property(
      TEST psp_${TARGET_NAME}
      APPEND
      PROPERTY DEPENDS prepare_${TARGET_NAME})
    set_property(
      TEST psp_${TARGET_NAME}
      APPEND
      PROPERTY FIXTURES_REQUIRED PrepareDecompilation_fixture_${TARGET_NAME})
  endif()

  # test VMA
  add_test(NAME vma_${TARGET_NAME}
           COMMAND revng --prefix=${CMAKE_BINARY_DIR} opt -vma
                   -vma-mincut-iter=50 ${OUTFOLDER}/prepared.bc -o /dev/null)
  set_property(
    TEST vma_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS prepare_${TARGET_NAME})
  set_property(
    TEST vma_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED PrepareDecompilation_fixture_${TARGET_NAME})
  set_property(
    TEST vma_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP VMA_fixture_${TARGET_NAME})

  # tests for IR Canonicalization
  add_test(
    NAME ir_canonicalization_${TARGET_NAME}
    COMMAND
      revng --prefix=${CMAKE_BINARY_DIR} opt -dla -make-model-gep -early-cse
      -dce -exit-ssa -twoscomplement-normalization ${OUTFOLDER}/prepared.bc -o
      ${OUTFOLDER}/canonicalized.bc)
  set_property(
    TEST ir_canonicalization_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS prepare_${TARGET_NAME})
  set_property(
    TEST ir_canonicalization_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED PrepareDecompilation_fixture_${TARGET_NAME})
  set_property(
    TEST ir_canonicalization_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP IRCanonicalization_fixture_${TARGET_NAME})

  # test revng-model-to-header
  add_test(
    NAME model_to_header_${TARGET_NAME}
    COMMAND revng --prefix . opt -model-to-header ${OUTFOLDER}/canonicalized.bc
            --types-header-name=${OUTFOLDER}/model_header.h -o /dev/null)
  set_property(
    TEST model_to_header_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS prepare_${TARGET_NAME})
  set_property(
    TEST model_to_header_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP ${TARGET_NAME}_model_header)
  set_property(
    TEST model_to_header_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED IRCanonicalization_fixture_${TARGET_NAME})
  # try to compile model headers without warnings
  add_test(NAME compile_model_header_${TARGET_NAME}
           COMMAND clang --config ${ASSETS_DIR}/compile_flags.txt
                   ${OUTFOLDER}/model_header.h -o /dev/null)
  set_property(
    TEST compile_model_header_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS model_to_header_${TARGET_NAME})
  set_property(
    TEST compile_model_header_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED ${TARGET_NAME}_model_header)

  # test revng-helpers-to-header
  add_test(
    NAME helpers_to_header_${TARGET_NAME}
    COMMAND
      revng --prefix . opt --helpers-to-header ${OUTFOLDER}/canonicalized.bc
      --helpers-header-name=${OUTFOLDER}/helpers_header.h -o /dev/null)
  set_property(
    TEST helpers_to_header_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP ${TARGET_NAME}_helpers_header)
  set_property(
    TEST helpers_to_header_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS prepare_${TARGET_NAME})
  set_property(
    TEST helpers_to_header_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED IRCanonicalization_fixture_${TARGET_NAME})
  # try to compile helpers headers without warnings
  add_test(NAME compile_helpers_header_${TARGET_NAME}
           COMMAND clang --config ${ASSETS_DIR}/compile_flags.txt
                   ${OUTFOLDER}/helpers_header.h -o /dev/null)
  set_property(
    TEST compile_helpers_header_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS helpers_to_header_${TARGET_NAME})
  set_property(
    TEST compile_helpers_header_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED ${TARGET_NAME}_helpers_header)

  # setup decompilation directory
  make_directory(${OUTFOLDER}/decompiled_stringbased)
  # copy recompilation assets to the decompilation directory
  file(COPY ${CMAKE_SOURCE_DIR}/${ASSETS_DIR}
       DESTINATION ${OUTFOLDER}/decompiled_stringbased)

  # test string-based decompiler
  add_test(
    NAME decompile_string_based_${TARGET_NAME}
    COMMAND
      revng --prefix=${CMAKE_BINARY_DIR} opt -S -restructure-cfg
      -add-ir-serialization-markers -beautify-ghast --helpers-to-header
      --helpers-header-name=${OUTFOLDER}/decompiled_stringbased/helpers.h
      -model-to-header
      --types-header-name=${OUTFOLDER}/decompiled_stringbased/revng-types.h
      -collect-local-vars -c-backend
      --c-decompiled-dir=${OUTFOLDER}/decompiled_stringbased
      ${OUTFOLDER}/canonicalized.bc --debug-log=c-backend-inline -o
      ${OUTFOLDER}/decompiled_stringbased/ir-during-decompilation.ll)
  set_property(
    TEST decompile_string_based_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS ir_canonicalization_${TARGET_NAME})
  set_property(
    TEST decompile_string_based_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED IRCanonicalization_fixture_${TARGET_NAME})
  set_property(
    TEST decompile_string_based_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP DecompiledStringBased_fixture_${TARGET_NAME})

  # test recompilation of the string-based decompiler
  add_test(
    NAME recompilation_string_based_${TARGET_NAME}
    COMMAND
      sh -c "for C_FILE in $(ls ${OUTFOLDER}/decompiled_stringbased/*.c); \
    do \
      clang \
      --config ${ASSETS_DIR}/compile_flags.txt \
      -o /dev/null \
      $C_FILE || exit -1; \
    done;")
  set_property(
    TEST recompilation_string_based_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS ir_canonicalization_${TARGET_NAME})
  set_property(
    TEST recompilation_string_based_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED DecompiledStringBased_fixture_${TARGET_NAME})
  set_property(
    TEST recompilation_string_based_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP RecompiledStringBased_fixture_${TARGET_NAME})

endmacro()

# Decompilation tests
macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)

  if(EXISTS ${INPUT_FILE})
    # Run decompilation tests on all analysis tests from revng and only on the
    # printf runtime tests from revng.
    if(("${INPUT_FILE}" MATCHES ".*/runtime/.*printf.*")
       OR ("${INPUT_FILE}" MATCHES ".*/analysis/.*"))
      add_decompilation_pipeline_test(${CATEGORY} ${INPUT_FILE}
                                      ${CONFIGURATION} ${OUTPUT} ${TARGET_NAME})
    endif()
  endif()

endmacro()

register_derived_artifact(
  "abi-enforced-for-decompilation" # FROM_ARTIFACTS: input artifacts
  "revng-c-passes" # NAME: name of the new aritfact
  "" # SUFFIX: extension of output file
  "FILE" # TYPE: "FILE" or "DIRECTORY"
)
