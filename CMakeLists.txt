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

# End-to-end tests for standalone passes
macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  message(VERBOSE "--------- Preparing test ---------")
  message(VERBOSE "Category: ${CATEGORY}")
  message(VERBOSE "Input file: ${INPUT_FILE}")
  message(VERBOSE "Configuration: ${CONFIGURATION}")
  message(VERBOSE "Output: ${OUTPUT}")
  message(VERBOSE "Target name: ${TARGET_NAME}")

  if (EXISTS ${INPUT_FILE})
    message(VERBOSE "Input file found: ${INPUT_FILE}")

    # prepare enforced .ll for decompilation
    add_test(NAME prepare_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRemoveCpuLoopStore.so
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRemoveExceptionCalls.so
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRemoveLLVMAssumeCalls.so
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRemoveNewPCCalls.so
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libMakeEnvNull.so
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libFilterForDecompilation.so
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/librevngTypeShrinking.so
      # Remove revng's artifacts from IR
      -filter-for-decompilation
      -remove-newpc-calls
      -remove-exception-calls
      -remove-cpu-loop-store
      -make-env-null
      # Optimize IR with LLVM's passes
      -dce
      -simplifycfg
      -dse
      -instcombine
      -sroa
      -constprop
      -jump-threading
      -licm
      -unreachableblockelim
      -instcombine
      -early-cse
      -simplifycfg
      -gvn-hoist
      # apply type shrinking
      -type-shrinking
      --min-width 8
      -early-cse
      -constprop
      -gvn
      -constprop
      -dse
      -dce
      # Remove LLVM's artifacts from IR
      -remove-llvmassume-calls
      -dce
      ${INPUT_FILE}
      -o ${TARGET_NAME}_prepared.ll
      )

    set_property(TEST prepare_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP PrepareDecompilation_fixture_${TARGET_NAME}
      )

    # promote stack pointer in .ll for decompilation
    add_test(NAME promote_stack_pointer_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libPromoteStackPointer.so
      -promote-stack-pointer
      ${TARGET_NAME}_prepared.ll
      -o ${TARGET_NAME}_promoted_stack.ll
      )
    set_property(TEST promote_stack_pointer_${TARGET_NAME} APPEND PROPERTY
      DEPENDS prepare_${TARGET_NAME}
      )
    set_property(TEST promote_stack_pointer_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED PrepareDecompilation_fixture_${TARGET_NAME}
      )
    set_property(TEST promote_stack_pointer_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP PromoteStackPointer_fixture_${TARGET_NAME}
      )

    # adjust stack pointer in .ll for decompilation
    add_test(NAME adjust_stack_pointer_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libAdjustStackPointer.so
      -sroa -instcombine -adjust-stack-pointer
      ${TARGET_NAME}_promoted_stack.ll
      -o ${TARGET_NAME}_adjusted_stack.ll
      )
    set_property(TEST adjust_stack_pointer_${TARGET_NAME} APPEND PROPERTY
      DEPENDS promote_stack_pointer_${TARGET_NAME}
      )
    set_property(TEST adjust_stack_pointer_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED PromoteStackPointer_fixture_${TARGET_NAME}
      )
    set_property(TEST adjust_stack_pointer_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP AdjustStackPointer_fixture_${TARGET_NAME}
      )

    # test DLA
    add_test(NAME dla_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libDecompiler.so
      -dla
      ${TARGET_NAME}_adjusted_stack.ll
      -o /dev/null
      )
    set_property(TEST dla_${TARGET_NAME} APPEND PROPERTY
      DEPENDS adjust_stack_pointer_${TARGET_NAME}
      )
    set_property(TEST dla_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED AdjustStackPointer_fixture_${TARGET_NAME}
      )
    set_property(TEST dla_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP DLA_fixture_${TARGET_NAME}
      )

    # test AddSCEVBarrierPass
    add_test(NAME add_scev_barrier_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libDecompiler.so
      -mark-for-serialization -add-scev-barrier
      ${TARGET_NAME}_adjusted_stack.ll
      -o /dev/null
      )
    set_property(TEST add_scev_barrier_${TARGET_NAME} APPEND PROPERTY
      DEPENDS adjust_stack_pointer_${TARGET_NAME}
      )
    set_property(TEST add_scev_barrier_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED AdjustStackPointer_fixture_${TARGET_NAME}
      )
    set_property(TEST add_scev_barrier_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP AddSCEVBarrier_fixture_${TARGET_NAME}
      )

    # test combing
    add_test(NAME comb_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRestructureCFGPass.so
      -restructure-cfg
      ${TARGET_NAME}_adjusted_stack.ll
      -o /dev/null
      )
    set_property(TEST comb_${TARGET_NAME} APPEND PROPERTY
      DEPENDS adjust_stack_pointer_${TARGET_NAME}
      )
    set_property(TEST comb_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED AdjustStackPointer_fixture_${TARGET_NAME}
      )
    set_property(TEST comb_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP Combed_fixture_${TARGET_NAME}
      )

    # test decompilation without DLA
    add_test(NAME decompilation_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRestructureCFGPass.so
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libDecompiler.so
      -restructure-cfg -decompilation
      ${TARGET_NAME}_adjusted_stack.ll
      -o /dev/null
      )
    set_property(TEST decompilation_${TARGET_NAME} APPEND PROPERTY
      DEPENDS comb_${TARGET_NAME}
      )
    set_property(TEST decompilation_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED Combed_fixture_${TARGET_NAME}
      )
    set_property(TEST decompilation_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP Decompiled_fixture_${TARGET_NAME}
      )

#    # test full decompilation pipeline with dla
#    add_test(NAME decompile_with_dla_${TARGET_NAME} COMMAND opt -S
#      -load
#      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libPHIASAPAssignmentInfo.so
#      -load
#      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRestructureCFGPass.so
#      -load
#      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libDecompiler.so
#      -dla
#      -phi-asap-assignment-info
#      -mark-for-serialization
#      -restructure-cfg
#      -add-scev-barrier
#      -decompilation
#      ${TARGET_NAME}_adjusted_stack.ll
#      -o /dev/null
#      )
#    set_property(TEST decompile_with_dla_${TARGET_NAME} APPEND PROPERTY
#      DEPENDS comb_${TARGET_NAME}
#      )
#    set_property(TEST decompile_with_dla_${TARGET_NAME} APPEND PROPERTY
#      FIXTURES_REQUIRED Combed_fixture_${TARGET_NAME}
#      )
#    set_property(TEST decompile_with_dla_${TARGET_NAME} APPEND PROPERTY
#      FIXTURES_REQUIRED DLA_fixture_${TARGET_NAME}
#      )
#    set_property(TEST decompile_with_dla_${TARGET_NAME} APPEND PROPERTY
#      FIXTURES_REQUIRED Decompiled_fixture_${TARGET_NAME}
#      )

  endif()
  message(VERBOSE "----------------------------------")
endmacro()

# Register a new artifact
register_derived_artifact("abi-enforced-for-decompilation" # FROM_ARTIFACTS: input artifacts
  "revng-c-passes"                 # NAME: name of the new aritfact
  ""                               # SUFFIX: extension of output file
  "FILE"                           # TYPE: "FILE" or "DIRECTORY"
  )

# Register a new artifact
register_derived_artifact("abi-enforced-for-decompilation-torture" # FROM_ARTIFACTS: input artifacts
  "revng-c-passes-torture"         # NAME: name of the new aritfact
  ""                               # SUFFIX: extension of output file
  "FILE"                           # TYPE: "FILE" or "DIRECTORY"
  )
