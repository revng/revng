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
  share/revng/pipelines/ir-canonicalization.yml
  share/revng/pipelines/opt-for-decompilation.yml
  share/revng/pipelines/promote-stack-pointer.yml
  share/revng/pipelines/remove-lifting-artifacts.yml
  share/revng/pipelines/segregate-stack-accesses.yml)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
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

  # Remove lifting artifacts
  add_test(
    NAME remove_lifting_artifacts_${TARGET_NAME}
    COMMAND
      revng --prefix=${CMAKE_BINARY_DIR} llvm pipeline "${INPUT_FILE}"
      "${OUTFOLDER}/lifted_artifacts_removed.bc" EnforceABI
      RemoveLiftingArtifacts "*:LiftingArtifactsRemoved")
  set_property(
    TEST remove_lifting_artifacts_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP remove_lifting_artifacts_fixture_${TARGET_NAME})

  # Promote stack pointer
  add_test(
    NAME promote_stack_pointer_${TARGET_NAME}
    COMMAND
      revng --prefix=${CMAKE_BINARY_DIR} llvm pipeline
      "${OUTFOLDER}/lifted_artifacts_removed.bc"
      "${OUTFOLDER}/promoted_stack_pointer.bc" RemoveLiftingArtifacts
      PromoteStackPointer "*:StackPointerPromoted")
  set_property(
    TEST promote_stack_pointer_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS remove_lifting_artifacts_${TARGET_NAME})
  set_property(
    TEST promote_stack_pointer_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED remove_lifting_artifacts_fixture_${TARGET_NAME})
  set_property(
    TEST promote_stack_pointer_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP promote_stack_pointer_fixture_${TARGET_NAME})

  # Optimize for decompilation and segregate stack accesses
  add_test(
    NAME optimize_and_slice_stack_${TARGET_NAME}
    COMMAND
      revng --prefix=${CMAKE_BINARY_DIR} llvm pipeline
      "${OUTFOLDER}/promoted_stack_pointer.bc"
      "${OUTFOLDER}/optimized_and_sliced_stack.bc" PromoteStackPointer
      LateOptimizeForDecompilation "*:StackAccessesSegregated")
  set_property(
    TEST optimize_and_slice_stack_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS promote_stack_pointer_${TARGET_NAME})
  set_property(
    TEST optimize_and_slice_stack_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED promote_stack_pointer_fixture_${TARGET_NAME})
  set_property(
    TEST optimize_and_slice_stack_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP optimize_and_slice_stack_fixture_${TARGET_NAME})

  get_filename_component(BASENAME "${INPUT_FILE}" NAME_WE)
  set(GROUND_TRUTH_MODEL
      "${CMAKE_SOURCE_DIR}/tests/Analysis/PromoteStackPointer/${BASENAME}.yml")
  if("${CATEGORY}" STREQUAL "tests_analysis_PromoteStackPointer"
     AND "${CONFIGURATION}" STREQUAL "dynamic_native"
     AND EXISTS "${GROUND_TRUTH_MODEL}")
    add_test(
      NAME psp_${TARGET_NAME}
      COMMAND
        sh -c
        "revng opt -S ${OUTFOLDER}/optimized_and_sliced_stack.bc | revng model dump | revng model compare - ${GROUND_TRUTH_MODEL}"
    )
    set_property(
      TEST psp_${TARGET_NAME}
      APPEND
      PROPERTY DEPENDS optimize_and_slice_stack_${TARGET_NAME})
    set_property(
      TEST psp_${TARGET_NAME}
      APPEND
      PROPERTY FIXTURES_REQUIRED
               optimize_and_slice_stack_fixture_${TARGET_NAME})
  endif()

  # test VMA
  add_test(
    NAME vma_${TARGET_NAME}
    COMMAND revng --prefix=${CMAKE_BINARY_DIR} opt -vma -vma-mincut-iter=50
            ${OUTFOLDER}/optimized_and_sliced_stack.bc -o /dev/null)
  set_property(
    TEST vma_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS optimize_and_slice_stack_${TARGET_NAME})
  set_property(
    TEST vma_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED optimize_and_slice_stack_fixture_${TARGET_NAME})
  set_property(
    TEST vma_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP VMA_fixture_${TARGET_NAME})

  # tests for IR Canonicalization
  add_test(
    NAME ir_canonicalization_${TARGET_NAME}
    COMMAND
      revng --prefix=${CMAKE_BINARY_DIR} opt -dla -make-model-gep -early-cse
      -dce -exit-ssa -twoscomplement-normalization -add-assignment-markers
      ${OUTFOLDER}/optimized_and_sliced_stack.bc -o
      ${OUTFOLDER}/canonicalized.bc)
  set_property(
    TEST ir_canonicalization_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS optimize_and_slice_stack_${TARGET_NAME})
  set_property(
    TEST ir_canonicalization_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED optimize_and_slice_stack_fixture_${TARGET_NAME})
  set_property(
    TEST ir_canonicalization_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP IRCanonicalization_fixture_${TARGET_NAME})

  # setup decompilation directory
  set(DECOMPILED_DIR ${OUTFOLDER}/decompiled)
  make_directory(${DECOMPILED_DIR})

  # test revng-model-to-header
  add_test(
    NAME model_to_header_${TARGET_NAME}
    COMMAND
      revng --prefix=${CMAKE_BINARY_DIR} model to-header
      -i=${OUTFOLDER}/canonicalized.bc -o=${DECOMPILED_DIR}/model_header.h)
  set_property(
    TEST model_to_header_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS optimize_and_slice_stack_${TARGET_NAME})
  set_property(
    TEST model_to_header_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP ${TARGET_NAME}_model_header)
  set_property(
    TEST model_to_header_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED IRCanonicalization_fixture_${TARGET_NAME})
  # try to compile model headers without warnings
  add_test(
    NAME compile_model_header_${TARGET_NAME}
    COMMAND clang --config ${CMAKE_BINARY_DIR}/${ASSETS_DIR}/compile_flags.txt
            ${DECOMPILED_DIR}/model_header.h -o /dev/null)
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
      revng --prefix=${CMAKE_BINARY_DIR} helpers-to-header
      -i=${OUTFOLDER}/canonicalized.bc -o=${DECOMPILED_DIR}/helpers_header.h)
  set_property(
    TEST helpers_to_header_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP ${TARGET_NAME}_helpers_header)
  set_property(
    TEST helpers_to_header_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS optimize_and_slice_stack_${TARGET_NAME})
  set_property(
    TEST helpers_to_header_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED IRCanonicalization_fixture_${TARGET_NAME})
  # try to compile helpers headers without warnings
  add_test(
    NAME compile_helpers_header_${TARGET_NAME}
    COMMAND clang --config ${CMAKE_BINARY_DIR}/${ASSETS_DIR}/compile_flags.txt
            ${DECOMPILED_DIR}/helpers_header.h -o /dev/null)
  set_property(
    TEST compile_helpers_header_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS helpers_to_header_${TARGET_NAME})
  set_property(
    TEST compile_helpers_header_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED ${TARGET_NAME}_helpers_header)

  # test the C decompiler backend
  add_test(
    NAME decompile_${TARGET_NAME}
    COMMAND
      revng --prefix=${CMAKE_BINARY_DIR} opt -S -restructure-cfg
      -beautify-ghast
      --backend-helpers-header-name=${DECOMPILED_DIR}/helpers_header.h
      --backend-types-header-name=${DECOMPILED_DIR}/model_header.h
      -collect-local-vars -c-backend --c-decompiled-dir=${DECOMPILED_DIR}
      ${OUTFOLDER}/canonicalized.bc -o /dev/null)
  set_property(
    TEST decompile_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS ir_canonicalization_${TARGET_NAME})
  set_property(
    TEST decompile_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED IRCanonicalization_fixture_${TARGET_NAME})
  set_property(
    TEST decompile_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP Decompiled_fixture_${TARGET_NAME})

  # test recompilation of C code generated by C decompiler backend
  add_test(
    NAME recompilation_${TARGET_NAME}
    COMMAND
      sh -c
      "cat ${DECOMPILED_DIR}/*.c | clang -x c - --config ${CMAKE_BINARY_DIR}/${ASSETS_DIR}/compile_flags.txt -o /dev/null "
  )
  set_property(
    TEST recompilation_${TARGET_NAME}
    APPEND
    PROPERTY DEPENDS ir_canonicalization_${TARGET_NAME})
  set_property(
    TEST recompilation_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_REQUIRED Decompiled_fixture_${TARGET_NAME})
  set_property(
    TEST recompilation_${TARGET_NAME}
    APPEND
    PROPERTY FIXTURES_SETUP Recompiled_fixture_${TARGET_NAME})

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
  "" # NAME: name of the new aritfact
  "" # SUFFIX: extension of output file
  "FILE" # TYPE: "FILE" or "DIRECTORY"
)

macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  get_filename_component(BASENAME "${INPUT_FILE}" NAME_WE)
  set(MODEL_OVERRIDE
      "${CMAKE_SOURCE_DIR}/tests/Analysis/SegregateStackAccesses/${BASENAME}.override.yml"
  )
  set(FILECHECK_FILE
      "${CMAKE_SOURCE_DIR}/tests/Analysis/SegregateStackAccesses/${BASENAME}.filecheck.ll"
  )

  if("${CATEGORY}" STREQUAL "tests_analysis_SegregateStackAccesses"
     AND "${CONFIGURATION}" STREQUAL "dynamic_native"
     AND EXISTS "${MODEL_OVERRIDE}"
     AND EXISTS "${FILECHECK_FILE}")
    set(REVNG "revng --prefix=${CMAKE_BINARY_DIR}")
    add_test(
      NAME segregate_stack_accesses_${TARGET_NAME}
      COMMAND
        bash -c
        "${REVNG} model override-by-name ${INPUT_FILE} ${MODEL_OVERRIDE} | ${REVNG} llvm pipeline - - Lift LateOptimizeForDecompilation *:StackAccessesSegregated | ${REVNG} opt -S | FileCheck ${FILECHECK_FILE}"
    )
  endif()

endmacro()
register_derived_artifact(
  "lifted" # FROM_ARTIFACTS: input artifacts
  "" # NAME: name of the new aritfact
  "" # SUFFIX: extension of output file
  "FILE" # TYPE: "FILE" or "DIRECTORY"
)
