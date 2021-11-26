#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

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

add_definitions("-DINSTALL_PATH=\"${CMAKE_INSTALL_PREFIX}\"")
add_definitions("-DBUILD_PATH=\"${CMAKE_BINARY_DIR}\"")

configure_file(recompile_dir.sh "${CMAKE_BINARY_DIR}/bin/recompile_dir")

include_directories(include/)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})

# LLVM CMake stuff
llvm_map_components_to_libnames(LLVM_LIBRARIES
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

add_subdirectory(include)
add_subdirectory(lib)

include(tests/Tests.cmake)

include(${CMAKE_INSTALL_PREFIX}/share/revng/qa/cmake/revng-qa.cmake)


# Macro for creating the tests for decompilation pipeline
macro(add_decompilation_pipeline_test CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)

    # test revng-model-to-header
    add_test(NAME model_to_header_${TARGET_NAME} COMMAND
      bash -c "${CMAKE_BINARY_DIR}/bin/revng-model-to-header < ${INPUT_FILE} > ${TARGET_NAME}_model_header.h"
      )
    set_property(TEST model_to_header_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP ${TARGET_NAME}_model_header
      )

    # prepare enforced .bc for decompilation
    add_test(NAME prepare_${TARGET_NAME} COMMAND revng --prefix=.
      opt
      # Remove revng's artifacts from IR
      -filter-for-decompilation
      -remove-newpc-calls
      -remove-exception-calls
      -remove-cpu-loop-store
      -remove-llvm-dbg-intrinsics
      -make-env-null
      # Stack promotion
      -measure-stack-size-at-call-sites
      -promote-stack-pointer
      # Optimize IR with LLVM's passes
      -dce
      -simplifycfg
      -dse
      -instcombine
      -sroa --sroa-aggressive-phis-selects
      -instsimplify
      -jump-threading
      -licm
      -unreachableblockelim
      -instcombine
      -early-cse
      -simplifycfg
      # apply type shrinking
      -type-shrinking
      --min-width 8
      -early-cse
      -instsimplify
      -gvn
      -instsimplify
      -dse
      -dce
      # Remove LLVM's artifacts from IR
      -remove-llvmassume-calls
      -dce
      # Stack analysis and segregation
      -instrument-stack-accesses
      -instcombine
      -compute-stack-accesses-bounds
      -detect-stack-size
      -segregate-stack-accesses
      -cleanup-stack-size-markers
      -dce
      -sroa
      -instcombine
      -sroa
      -simplifycfg
      ${INPUT_FILE}
      -o ${TARGET_NAME}_prepared.bc
      )
    set_property(TEST prepare_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP PrepareDecompilation_fixture_${TARGET_NAME}
      )

    # test VMA
    add_test(NAME vma_${TARGET_NAME} COMMAND revng --prefix=.
      opt
      -vma
      -vma-mincut-iter=50
      ${TARGET_NAME}_prepared.bc
      -o /dev/null
      )
    set_property(TEST vma_${TARGET_NAME} APPEND PROPERTY
      DEPENDS prepare_${TARGET_NAME}
      )
    set_property(TEST vma_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED PrepareDecompilation_fixture_${TARGET_NAME}
      )
    set_property(TEST vma_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP VMA_fixture_${TARGET_NAME}
      )

    # test combing and decompilation without DLA
    add_test(NAME decompilation_${TARGET_NAME} COMMAND revng --prefix=.
      opt
      -restructure-cfg
      -beautify-ghast
      -decompilation
      -decompiled-dir=${TARGET_NAME}_decompiled_dir
      ${TARGET_NAME}_prepared.bc
      -o /dev/null
      )
    set_property(TEST decompilation_${TARGET_NAME} APPEND PROPERTY
      DEPENDS prepare_${TARGET_NAME}
      )
    set_property(TEST decompilation_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED PrepareDecompilation_fixture_${TARGET_NAME}
      )
    set_property(TEST decompilation_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP Decompiled_fixture_${TARGET_NAME}
      )

    # test recompilation
    add_test(NAME recompilation_${TARGET_NAME} COMMAND
      ${CMAKE_BINARY_DIR}/bin/recompile_dir ${TARGET_NAME}_decompiled_dir
      )
    set_property(TEST recompilation_${TARGET_NAME} APPEND PROPERTY
      DEPENDS prepare_${TARGET_NAME}
      )
    set_property(TEST recompilation_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED Decompiled_fixture_${TARGET_NAME}
      )
    set_property(TEST recompilation_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP Recompiled_fixture_${TARGET_NAME}
      )

    # test full pipeline with dla combing and decompilation
    add_test(NAME decompile_with_dla_${TARGET_NAME} COMMAND revng --prefix=.
      opt
      -dla
      -phi-asap-assignment-info
      -restructure-cfg
      -beautify-ghast
      -mark-for-serialization
      -add-scev-barrier
      -decompilation
      -dla-flatc-dir=${TARGET_NAME}_dla_dir
      ${TARGET_NAME}_prepared.bc
      -o /dev/null
      )
    set_property(TEST decompile_with_dla_${TARGET_NAME} APPEND PROPERTY
      DEPENDS prepare_${TARGET_NAME}
      )
    set_property(TEST decompile_with_dla_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED PrepareDecompilation_fixture_${TARGET_NAME}
      )

endmacro()


# Decompilation torture tests
macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)

  if (EXISTS ${INPUT_FILE})
    add_decompilation_pipeline_test(${CATEGORY}
                                    ${INPUT_FILE}
                                    ${CONFIGURATION}
                                    ${OUTPUT}
                                    ${TARGET_NAME})
  endif()

endmacro()

register_derived_artifact("abi-enforced-for-decompilation-torture" # FROM_ARTIFACTS: input artifacts
  "revng-c-passes-torture"         # NAME: name of the new aritfact
  ""                               # SUFFIX: extension of output file
  "FILE"                           # TYPE: "FILE" or "DIRECTORY"
  )


# Decompilation tests
macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)

  if (EXISTS ${INPUT_FILE})
    # Run decompilation tests on all analysis tests from revng and only on
    # the printf runtime tests from revng.
    if (("${INPUT_FILE}" MATCHES ".*/runtime/.*printf.*")
         OR ("${INPUT_FILE}" MATCHES ".*/analysis/.*"))
      add_decompilation_pipeline_test(${CATEGORY}
                                      ${INPUT_FILE}
                                      ${CONFIGURATION}
                                      ${OUTPUT}
                                      ${TARGET_NAME})
    endif()
  endif()

endmacro()

register_derived_artifact("abi-enforced-for-decompilation" # FROM_ARTIFACTS: input artifacts
  "revng-c-passes"                 # NAME: name of the new aritfact
  ""                               # SUFFIX: extension of output file
  "FILE"                           # TYPE: "FILE" or "DIRECTORY"
  )
