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

    add_test(NAME remove_newpc_calls_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRemoveNewPCCalls.so
      -remove-newpc-calls
      ${INPUT_FILE}
      -o /dev/null
      )
    set_property(TEST remove_newpc_calls_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP RemoveNewPCCalls_fixture_${TARGET_NAME}
      )

    add_test(NAME remove_exception_calls_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRemoveExceptionCalls.so
      -remove-exception-calls
      ${INPUT_FILE}
      -o /dev/null
      )
    set_property(TEST remove_exception_calls_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP RemoveExceptionCalls_fixture_${TARGET_NAME}
      )

    add_test(NAME remove_llvmassume_calls_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRemoveLLVMAssumeCalls.so
      -remove-llvmassume-calls
      ${INPUT_FILE}
      -o /dev/null
      )
    set_property(TEST remove_llvmassume_calls_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP RemoveLLVMAssumeCalls_fixture_${TARGET_NAME}
      )

    add_test(NAME remove_cpuloop_calls_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRemoveCpuLoopStore.so
      -remove-cpu-loop-store
      ${INPUT_FILE}
      -o /dev/null
      )
    set_property(TEST remove_cpuloop_calls_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP RemoveCpuLoopCalls_fixture_${TARGET_NAME}
      )

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
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRemovePCStoresPass.so
      # Remove revng's artifacts from IR
      -remove-newpc-calls
      -remove-exception-calls
      -remove-cpu-loop-store
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
      # Remove LLVM's artifacts from IR
      -remove-llvmassume-calls
      -dce
      ${INPUT_FILE}
      -o ${TARGET_NAME}_prepared.ll
      )

    set_property(TEST prepare_${TARGET_NAME} APPEND PROPERTY
      DEPENDS remove_newpc_calls_${TARGET_NAME}
      )
    set_property(TEST prepare_${TARGET_NAME} APPEND PROPERTY
      DEPENDS remove_exception_calls_${TARGET_NAME}
      )
    set_property(TEST prepare_${TARGET_NAME} APPEND PROPERTY
      DEPENDS remove_llvmassume_calls_${TARGET_NAME}
      )
    set_property(TEST prepare_${TARGET_NAME} APPEND PROPERTY
      DEPENDS remove_cpuloop_calls_${TARGET_NAME}
      )
    set_property(TEST prepare_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED RemoveNewPCCalls_fixture_${TARGET_NAME}
      )
    set_property(TEST prepare_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED RemoveExceptionCalls_fixture_${TARGET_NAME}
      )
    set_property(TEST prepare_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED RemoveLLVMAssumeCalls_fixture_${TARGET_NAME}
      )
    set_property(TEST prepare_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED RemoveCpuLoopCalls_fixture_${TARGET_NAME}
      )
    set_property(TEST prepare_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP PrepareDecompilation_fixture_${TARGET_NAME}
      )

    # test combing
    add_test(NAME comb_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libRestructureCFGPass.so
      -restructure-cfg
      ${TARGET_NAME}_prepared.ll
      -o /dev/null
      )
    set_property(TEST comb_${TARGET_NAME} APPEND PROPERTY
      DEPENDS prepare_${TARGET_NAME}
      )
    set_property(TEST comb_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED PrepareDecompilation_fixture_${TARGET_NAME}
      )
    set_property(TEST comb_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_SETUP Combed_fixture_${TARGET_NAME}
      )

    # test decompilation
    add_test(NAME decompilation_${TARGET_NAME} COMMAND opt -S
      -load
      ${CMAKE_CURRENT_BINARY_DIR}/lib/revng/analyses/libDecompiler.so
      -decompilation
      ${TARGET_NAME}_prepared.ll
      -o /dev/null
      )
    set_property(TEST decompilation_${TARGET_NAME} APPEND PROPERTY
      DEPENDS comb_${TARGET_NAME}
      )
    set_property(TEST decompilation_${TARGET_NAME} APPEND PROPERTY
      FIXTURES_REQUIRED Combed_fixture_${TARGET_NAME}
      )

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
