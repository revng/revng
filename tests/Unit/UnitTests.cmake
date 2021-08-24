#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

cmake_policy(SET CMP0060 NEW)

include(${CMAKE_INSTALL_PREFIX}/share/revng/qa/cmake/revng-qa.cmake)

set(SRC "${CMAKE_SOURCE_DIR}/tests/Unit")

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

#
# reachability_library
#
include(${SRC}/Reachability/Reachability.cmake)

#
# test_reachabilitypass
#

revng_add_private_executable(test_reachabilitypass "${SRC}/ReachabilityPass.cpp")
target_compile_definitions(test_reachabilitypass
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_reachabilitypass
  PRIVATE "${CMAKE_SOURCE_DIR}"
  "${Boost_INCLUDE_DIRS}")
target_link_libraries(test_reachabilitypass
  Reachability
  revng::revngModel
  revng::revngSupport
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_reachabilitypass COMMAND ./bin/test_reachabilitypass)

#
# test_combingpass
#

revng_add_private_executable(test_combingpass "${SRC}/CombingPass.cpp")
target_compile_definitions(test_combingpass
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_combingpass
  PRIVATE "${CMAKE_SOURCE_DIR}"
  "${Boost_INCLUDE_DIRS}")
target_link_libraries(test_combingpass
  RestructureCFGPass
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_combingpass COMMAND ./bin/test_combingpass -- "${SRC}/TestGraphs/")

#
# test_vma
#

revng_add_private_executable(test_vma "${SRC}/ValueManipulationAnalysis.cpp")
target_compile_definitions(test_vma
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_vma
  PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_vma
  ValueManipulationAnalysis
  revng::revngModel
  revng::revngSupport
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_vma COMMAND ./bin/test_vma)
set_tests_properties(test_vma PROPERTIES LABELS "unit")


revng_add_private_executable(decompile_function "${SRC}/DecompileFunction.cpp")
target_include_directories(decompile_function
  PRIVATE "${CMAKE_SOURCE_DIR}"
          "${Boost_INCLUDE_DIRS}")
target_compile_definitions(decompile_function
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(decompile_function
  Decompiler
  revng::revngModel
  revng::revngSupport
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})

# End-to-end tests for the decompilation pipeline public API decompileFunction
macro(artifact_handler CATEGORY INPUT_FILE CONFIGURATION OUTPUT TARGET_NAME)
  if (EXISTS ${INPUT_FILE})
    add_test(NAME decompile_function_${TARGET_NAME} COMMAND ./bin/decompile_function ${INPUT_FILE})
  endif()
endmacro()

# Register a new artifact
register_derived_artifact("abi-enforced-for-decompilation-torture" # FROM_ARTIFACTS: input artifacts
  "decompilation-pipeline-artifact"         # NAME: name of the new aritfact
  ""                               # SUFFIX: extension of output file
  "FILE"                           # TYPE: "FILE" or "DIRECTORY"
  )

#
# dla_step_manager
#

revng_add_private_executable(dla_step_manager "${SRC}/DLAStepManager.cpp")
target_compile_definitions(dla_step_manager
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(dla_step_manager
  PRIVATE "${CMAKE_SOURCE_DIR}"
  "${Boost_INCLUDE_DIRS}")
target_link_libraries(dla_step_manager
  Decompiler
  clangSerialization
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME dla_step_manager COMMAND ./bin/dla_step_manager)

#
# MarkForSerializationTest
#

revng_add_private_executable(MarkForSerializationTest
  "${SRC}/MarkForSerializationTest.cpp")
target_compile_definitions(MarkForSerializationTest
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(MarkForSerializationTest
  PRIVATE "${CMAKE_SOURCE_DIR}"
  "${Boost_INCLUDE_DIRS}")
target_link_libraries(MarkForSerializationTest
  Decompiler
  clangSerialization
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME MarkForSerializationTest COMMAND ./bin/MarkForSerializationTest)

#
# DLACollapseSingleChild
#

revng_add_private_executable(dla_steps "${SRC}/DLASteps.cpp")
target_compile_definitions(dla_steps
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(dla_steps
  PRIVATE "${CMAKE_SOURCE_DIR}"
  "${Boost_INCLUDE_DIRS}")
target_link_libraries(dla_steps
  Decompiler
  DataLayoutAnalysis
  revng::revngModel
  revng::revngSupport
  revng::revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME dla_steps COMMAND ./bin/dla_steps)
