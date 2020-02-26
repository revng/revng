#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_policy(SET CMP0060 NEW)

set(SRC "${CMAKE_SOURCE_DIR}/tests/unit")

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

#
# test_lazysmallbitvector
#

revng_add_private_executable(test_lazysmallbitvector "${SRC}/LazySmallBitVector.cpp")
target_compile_definitions(test_lazysmallbitvector
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_lazysmallbitvector
  PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_lazysmallbitvector
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_lazysmallbitvector COMMAND ./bin/test_lazysmallbitvector)
set_tests_properties(test_lazysmallbitvector PROPERTIES LABELS "unit")

#
# test_stackanalysis
#

revng_add_private_executable(test_stackanalysis "${SRC}/StackAnalysis.cpp")
target_compile_definitions(test_stackanalysis
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_stackanalysis
  PRIVATE "${CMAKE_SOURCE_DIR}"
          "${CMAKE_SOURCE_DIR}/lib/StackAnalysis"
          "${CMAKE_BINARY_DIR}/lib/StackAnalysis")
target_link_libraries(test_stackanalysis
  revngStackAnalysis
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_stackanalysis COMMAND ./bin/test_stackanalysis)
set_tests_properties(test_stackanalysis PROPERTIES LABELS "unit")

#
# test_classsentinel
#

revng_add_private_executable(test_classsentinel "${SRC}/ClassSentinel.cpp")
target_compile_definitions(test_classsentinel
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_classsentinel
  PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_classsentinel
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_classsentinel COMMAND ./bin/test_classsentinel)
set_tests_properties(test_classsentinel PROPERTIES LABELS "unit")

#
# test_irhelpers
#

revng_add_private_executable(test_irhelpers "${SRC}/IRHelpers.cpp")
target_compile_definitions(test_irhelpers
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_irhelpers
  PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_irhelpers
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_irhelpers COMMAND ./bin/test_irhelpers)
set_tests_properties(test_irhelpers PROPERTIES LABELS "unit")

#
# test_irhelpers
#

revng_add_private_executable(test_advancedvalueinfo "${SRC}/AdvancedValueInfo.cpp")
target_compile_definitions(test_advancedvalueinfo
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_advancedvalueinfo
  PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_advancedvalueinfo
  revngSupport
  revngBasicAnalyses
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_advancedvalueinfo COMMAND ./bin/test_advancedvalueinfo)
set_tests_properties(test_advancedvalueinfo PROPERTIES LABELS "unit")

#
# test_zipmapiterator
#

revng_add_private_executable(test_zipmapiterator "${SRC}/ZipMapIterator.cpp")
target_compile_definitions(test_zipmapiterator
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_zipmapiterator
  PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_zipmapiterator
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_zipmapiterator COMMAND ./bin/test_zipmapiterator)
set_tests_properties(test_zipmapiterator PROPERTIES LABELS "unit")

#
# test_constantrangeset
#

revng_add_private_executable(test_constantrangeset "${SRC}/ConstantRangeSet.cpp")
target_compile_definitions(test_constantrangeset
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_constantrangeset
  PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_constantrangeset
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_constantrangeset COMMAND ./bin/test_constantrangeset)
set_tests_properties(test_constantrangeset PROPERTIES LABELS "unit")

#
# test_shrinkinstructionoperands
#

revng_add_private_executable(test_shrinkinstructionoperands "${SRC}/ShrinkInstructionOperandsPass.cpp")
target_compile_definitions(test_shrinkinstructionoperands
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_shrinkinstructionoperands
  PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_shrinkinstructionoperands
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_shrinkinstructionoperands COMMAND ./bin/test_shrinkinstructionoperands)
set_tests_properties(test_shrinkinstructionoperands PROPERTIES LABELS "unit")

#
# test_metaaddress
#

revng_add_private_executable(test_metaaddress "${SRC}/MetaAddress.cpp")
target_include_directories(test_metaaddress
  PRIVATE "${CMAKE_SOURCE_DIR}"
          "${Boost_INCLUDE_DIRS}")
target_compile_definitions(test_metaaddress
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(test_metaaddress
  revngSupport
  revngUnitTestHelpers
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  ${LLVM_LIBRARIES})
add_test(NAME test_metaaddress COMMAND ./bin/test_metaaddress)
set_tests_properties(test_metaaddress PROPERTIES LABELS "unit")

#
# test_filtered_graph_traits
#

revng_add_private_executable(test_filtered_graph_traits "${SRC}/FilteredGraphTraits.cpp")
target_compile_definitions(test_filtered_graph_traits
  PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_filtered_graph_traits
  PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_filtered_graph_traits
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_filtered_graph_traits COMMAND ./bin/test_filtered_graph_traits -- "${SRC}/test_graphs/")
set_tests_properties(test_filtered_graph_traits PROPERTIES LABELS "unit")
