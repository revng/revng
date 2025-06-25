#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

cmake_policy(SET CMP0060 NEW)

set(SRC "${CMAKE_SOURCE_DIR}/tests/unit")

add_subdirectory(llvm_lit_tests)
add_subdirectory(mlir_lit_tests)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

#
# test_lazysmallbitvector
#

revng_add_test_executable(test_lazysmallbitvector
                          "${SRC}/LazySmallBitVector.cpp")
target_compile_definitions(test_lazysmallbitvector
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_lazysmallbitvector
                           PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_lazysmallbitvector revngSupport revngUnitTestHelpers
                      Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_lazysmallbitvector COMMAND test_lazysmallbitvector)
set_tests_properties(test_lazysmallbitvector PROPERTIES LABELS "unit")

#
# test_classsentinel
#

revng_add_test_executable(test_classsentinel "${SRC}/ClassSentinel.cpp")
target_compile_definitions(test_classsentinel PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_classsentinel PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_classsentinel revngSupport revngUnitTestHelpers
                      Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_classsentinel COMMAND test_classsentinel)
set_tests_properties(test_classsentinel PROPERTIES LABELS "unit")

#
# test_irhelpers
#

revng_add_test_executable(test_irhelpers "${SRC}/IRHelpers.cpp")
target_compile_definitions(test_irhelpers PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_irhelpers PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_irhelpers revngSupport revngUnitTestHelpers
                      Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_irhelpers COMMAND test_irhelpers)
set_tests_properties(test_irhelpers PROPERTIES LABELS "unit")

#
# test_advancedvalueinfo
#

revng_add_test_executable(test_advancedvalueinfo "${SRC}/AdvancedValueInfo.cpp")
target_compile_definitions(test_advancedvalueinfo
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_advancedvalueinfo PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_advancedvalueinfo revngSupport revngBasicAnalyses revngValueMaterializer
  Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_advancedvalueinfo COMMAND test_advancedvalueinfo)
set_tests_properties(test_advancedvalueinfo PROPERTIES LABELS "unit")

#
# test_zipmapiterator
#

revng_add_test_executable(test_zipmapiterator "${SRC}/ZipMapIterator.cpp")
target_compile_definitions(test_zipmapiterator PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_zipmapiterator PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_zipmapiterator revngSupport revngUnitTestHelpers
                      Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_zipmapiterator COMMAND test_zipmapiterator)
set_tests_properties(test_zipmapiterator PROPERTIES LABELS "unit")

#
# test_constantrangeset
#

revng_add_test_executable(test_constantrangeset "${SRC}/ConstantRangeSet.cpp")
target_compile_definitions(test_constantrangeset
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_constantrangeset PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_constantrangeset revngSupport revngUnitTestHelpers
                      Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_constantrangeset COMMAND test_constantrangeset)
set_tests_properties(test_constantrangeset PROPERTIES LABELS "unit")

#
# test_shrinkinstructionoperands
#

revng_add_test_executable(test_shrinkinstructionoperands
                          "${SRC}/ShrinkInstructionOperandsPass.cpp")
target_compile_definitions(test_shrinkinstructionoperands
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_shrinkinstructionoperands
                           PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_shrinkinstructionoperands revngUnitTestHelpers revngBasicAnalyses
  Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_shrinkinstructionoperands COMMAND
               test_shrinkinstructionoperands)
set_tests_properties(test_shrinkinstructionoperands PROPERTIES LABELS "unit")

#
# test_metaaddress
#

revng_add_test_executable(test_metaaddress "${SRC}/MetaAddress.cpp")
target_include_directories(test_metaaddress PRIVATE "${CMAKE_SOURCE_DIR}"
                                                    "${Boost_INCLUDE_DIRS}")
target_compile_definitions(test_metaaddress PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(test_metaaddress revngSupport revngUnitTestHelpers
                      ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${LLVM_LIBRARIES})
revng_add_test(NAME test_metaaddress COMMAND test_metaaddress)
set_tests_properties(test_metaaddress PROPERTIES LABELS "unit")

#
# test_filtered_graph_traits
#

revng_add_test_executable(test_filtered_graph_traits
                          "${SRC}/FilteredGraphTraits.cpp")
target_compile_definitions(test_filtered_graph_traits
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_filtered_graph_traits
                           PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_filtered_graph_traits revngSupport revngUnitTestHelpers
  Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_filtered_graph_traits COMMAND
               test_filtered_graph_traits -- "${SRC}/test_graphs/")
set_tests_properties(test_filtered_graph_traits PROPERTIES LABELS "unit")

#
# test_smallmap
#

revng_add_test_executable(test_smallmap "${SRC}/SmallMap.cpp")
target_compile_definitions(test_smallmap PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_smallmap PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_smallmap revngSupport revngUnitTestHelpers
                      Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_smallmap COMMAND test_smallmap)
set_tests_properties(test_smallmap PROPERTIES LABELS "unit")

#
# test_genericgraph
#

revng_add_test_executable(test_genericgraph "${SRC}/GenericGraph.cpp")
target_compile_definitions(test_genericgraph PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_genericgraph PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_genericgraph revngSupport Boost::unit_test_framework
                      ${LLVM_LIBRARIES})
revng_add_test(NAME test_genericgraph COMMAND test_genericgraph)
set_tests_properties(test_genericgraph PROPERTIES LABELS "unit")

#
# test_graphalgorithms
#

revng_add_test_executable(test_graphalgorithms "${SRC}/GraphAlgorithms.cpp")
target_compile_definitions(test_graphalgorithms PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_graphalgorithms PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_graphalgorithms revngSupport
                      Boost::unit_test_framework ${LLVM_LIBRARIES})
add_test(NAME test_graphalgorithms COMMAND test_graphalgorithms)
set_tests_properties(test_graphalgorithms PROPERTIES LABELS "unit")

#
# test_keyedobjectscontainers
#

revng_add_test_executable(test_keyedobjectscontainers
                          "${SRC}/KeyedObjectsContainers.cpp")
target_compile_definitions(test_keyedobjectscontainers
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_keyedobjectscontainers
                           PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_keyedobjectscontainers revngSupport revngUnitTestHelpers revngModel
  Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_keyedobjectscontainers COMMAND
               test_keyedobjectscontainers)
set_tests_properties(test_keyedobjectscontainers PROPERTIES LABELS "unit")

#
# test_model
#

revng_add_test_executable(test_model "${SRC}/Model.cpp")
target_compile_definitions(test_model PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_model PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_model
  revngSupport
  revngUnitTestHelpers
  revngModel
  revngModelPasses
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
revng_add_test(NAME test_model COMMAND test_model)
set_tests_properties(test_model PROPERTIES LABELS "unit")

#
# test_instantiatepasses
#

revng_add_test_executable(test_instantiatepasses "${SRC}/InstantiatePasses.cpp")
target_compile_definitions(test_instantiatepasses
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_instantiatepasses PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_instantiatepasses revngSupport revngUnitTestHelpers
                      revngModel Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_instantiatepasses COMMAND test_instantiatepasses)
set_tests_properties(test_instantiatepasses PROPERTIES LABELS "unit")

#
# test_upcastablepointer
#

revng_add_test_executable(test_upcastablepointer "${SRC}/UpcastablePointer.cpp")
target_compile_definitions(test_upcastablepointer
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_upcastablepointer PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_upcastablepointer revngSupport revngUnitTestHelpers
                      revngModel Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_upcastablepointer COMMAND test_upcastablepointer)
set_tests_properties(test_upcastablepointer PROPERTIES LABELS "unit")

#
# test_recursive_coroutines
#

macro(add_recursive_coroutine_test NAME)
  revng_add_test_executable("${NAME}" "${SRC}/RecursiveCoroutine.cpp")
  target_compile_definitions("${NAME}" PRIVATE "BOOST_TEST_DYN_LINK=1")
  target_include_directories("${NAME}" PRIVATE "${CMAKE_SOURCE_DIR}")
  target_link_libraries("${NAME}" revngSupport ${LLVM_LIBRARIES})
  revng_add_test(NAME "${NAME}" COMMAND "${NAME}")
  set_tests_properties("${NAME}" PROPERTIES LABELS "unit")
endmacro()

add_recursive_coroutine_test(test_recursive_coroutines)

add_recursive_coroutine_test(test_recursive_coroutines_fallback)
target_compile_definitions(test_recursive_coroutines_fallback
                           PRIVATE DISABLE_RECURSIVE_COROUTINES)

add_recursive_coroutine_test(test_recursive_coroutines_iterative)
target_compile_definitions(test_recursive_coroutines_iterative
                           PRIVATE ITERATIVE)

#
# test_model_type
#

revng_add_test_executable(test_model_type "${SRC}/ModelType.cpp")
target_compile_definitions(test_model_type PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_model_type PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_model_type revngSupport revngUnitTestHelpers
                      revngModel Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_model_type COMMAND test_model_type)
set_tests_properties(test_model_type PROPERTIES LABELS "unit")

#
# test_tuple_tree_generator
#

set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/tests/unit/TupleTreeGenerator")
set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/tests/unit/TupleTreeGenerator")
set(HEADERS_DIR "${OUTPUT_DIR}/include")
set(CPP_DIR "${OUTPUT_DIR}/lib")

revng_add_test_executable(test_tuple_tree_generator
                          "${SRC}/TupleTreeGenerator/Test.cpp")
target_include_directories(test_tuple_tree_generator PUBLIC "${HEADERS_DIR}"
                                                            "${INCLUDE_DIR}")
target_tuple_tree_generator(
  test_tuple_tree_generator
  NAMESPACE
  ttgtest
  SCHEMA_PATH
  "${CMAKE_SOURCE_DIR}/tests/unit/TupleTreeGenerator/schema.yml"
  HEADERS_PATH
  "${HEADERS_DIR}/Generated"
  INCLUDE_PATH_PREFIX
  ".")
target_compile_definitions(test_tuple_tree_generator
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_link_libraries(test_tuple_tree_generator revngUnitTestHelpers revngModel
                      Boost::unit_test_framework)

revng_add_test(NAME test_tuple_tree_generator COMMAND test_tuple_tree_generator)
set_tests_properties(test_tuple_tree_generator PROPERTIES LABELS "unit")

#
# test_pipeline
#

revng_add_test_executable(test_pipeline "${SRC}/Pipeline.cpp")
target_compile_definitions(test_pipeline PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_pipeline PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_pipeline revngUnitTestHelpers revngPipeline
                      Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_pipeline COMMAND test_pipeline)
set_tests_properties(test_pipeline PROPERTIES LABELS "unit")

#
# test_pipeline_c
#

revng_add_test_executable(test_pipeline_c "${SRC}/PipelineC.cpp")
target_compile_definitions(test_pipeline_c PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_pipeline_c PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_pipeline_c
  revngSupport
  revngUnitTestHelpers
  revngPipelineC
  revngStringContainerLibrary
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
revng_add_test(NAME test_pipeline_c COMMAND test_pipeline_c)
set_tests_properties(test_pipeline_c PROPERTIES LABELS "unit")

#
# test_register_state_deductions
#

revng_add_test_executable(test_register_state_deductions
                          "${SRC}/RegisterStateDeductions.cpp")
target_compile_definitions(test_register_state_deductions
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_register_state_deductions
                           PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_register_state_deductions
  revngABI
  revngModel
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
revng_add_test(NAME test_register_state_deductions COMMAND
               test_register_state_deductions)
set_tests_properties(test_register_state_deductions PROPERTIES LABELS
                                                               "unit;abi")

#
# test_location_infrastructure
#

revng_add_test_executable(test_location_infrastructure "${SRC}/Location.cpp")
target_compile_definitions(test_location_infrastructure
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_location_infrastructure
                           PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_location_infrastructure
  revngModel
  revngPipeline
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
revng_add_test(NAME test_location_infrastructure COMMAND
               test_location_infrastructure)
set_tests_properties(test_location_infrastructure
                     PROPERTIES LABELS "unit;pipeline;location")

#
# test_alignment
#

revng_add_test_executable(test_alignment "${SRC}/Alignment.cpp")
target_compile_definitions(test_alignment PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_alignment PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_alignment
  revngABI
  revngModel
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
revng_add_test(NAME test_alignment COMMAND test_alignment)
set_tests_properties(test_alignment PROPERTIES LABELS "unit;abi;alignment")

#
# test_pipeline_c_tracing
#

configure_file("${CMAKE_SOURCE_DIR}/tests/pipeline/CopyPipeTestPipeline.yml"
               "${CMAKE_BINARY_DIR}/tests/pipeline/Tracing.yml" COPYONLY)

revng_add_test_executable(test_pipeline_c_tracing "PipelineCTracing.cpp")
target_compile_definitions(test_pipeline_c_tracing
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_pipeline_c_tracing
                           PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_pipeline_c_tracing revngPipelineC revngSupport
                      Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_pipeline_c_tracing COMMAND test_pipeline_c_tracing)
set_tests_properties(test_pipeline_c_tracing PROPERTIES LABELS "unit")

#
# test_gzip_tar_file
#

revng_add_test_executable(test_gzip_tar_file "${SRC}/GzipTarFile.cpp")
target_compile_definitions(test_gzip_tar_file PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_gzip_tar_file PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_gzip_tar_file revngSupport revngUnitTestHelpers
                      Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_gzip_tar_file COMMAND test_gzip_tar_file)
set_tests_properties(test_gzip_tar_file PROPERTIES LABELS "unit")

revng_add_test_executable(test_gzip_tar_fileGenerator
                          "${SRC}/GzipTarFileGenerator.cpp")
target_link_libraries(test_gzip_tar_fileGenerator revngSupport
                      ${LLVM_LIBRARIES})
revng_add_test(
  NAME test_gzip_tar_fileGenerator COMMAND
  "${CMAKE_CURRENT_SOURCE_DIR}/GzipTarFileGeneratorTest.sh"
  "${CMAKE_BINARY_DIR}")
set_tests_properties(test_gzip_tar_fileGenerator PROPERTIES LABELS "unit")

#
# test_type_bucket
#

revng_add_test_executable(test_type_bucket "${SRC}/TypeBucket.cpp")
target_compile_definitions(test_type_bucket PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_type_bucket PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_type_bucket revngModel revngSupport revngUnitTestHelpers
  Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_type_bucket COMMAND test_type_bucket)
set_tests_properties(test_type_bucket PROPERTIES LABELS
                                                 "unit;model;type_bucket")

#
# test_register_usage_analysis
#

revng_add_test_executable(test_register_usage_analysis
                          "${SRC}/RegisterUsageAnalyses.cpp")
target_compile_definitions(test_register_usage_analysis
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_register_usage_analysis
                           PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_register_usage_analysis revngSupport revngUnitTestHelpers
  Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_register_usage_analysis COMMAND
               test_register_usage_analysis)
set_tests_properties(test_register_usage_analysis
                     PROPERTIES LABELS "unit;model;type_bucket")

#
# test_adt
#

revng_add_test_executable(test_adt "${SRC}/ADT.cpp")
target_compile_definitions(test_adt PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_adt PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(test_adt revngSupport revngUnitTestHelpers
                      Boost::unit_test_framework ${LLVM_LIBRARIES})
revng_add_test(NAME test_adt COMMAND test_adt)
set_tests_properties(test_adt PROPERTIES LABELS "unit")

#
# test_type_shrinking
#
revng_add_test(
  NAME
  test_type_shrinking
  COMMAND
  bash
  -c
  "${CMAKE_BINARY_DIR}/bin/revng opt -S -early-type-shrinking -type-shrinking -instcombine ${SRC}/TypeShrinking.ll | FileCheck ${SRC}/TypeShrinking.ll"
)

#
# test_combingpass
#

revng_add_test_executable(test_combingpass "${SRC}/CombingPass.cpp")
target_compile_definitions(test_combingpass PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_combingpass PRIVATE "${CMAKE_SOURCE_DIR}"
                                                    "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  test_combingpass
  revngRestructureCFG
  revngModel
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_combingpass COMMAND test_combingpass -- "${SRC}/TestGraphs/")

#
# test_dla_step_manager
#

revng_add_test_executable(test_dla_step_manager "${SRC}/DLAStepManager.cpp")
target_compile_definitions(test_dla_step_manager
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(
  test_dla_step_manager PRIVATE "${CMAKE_SOURCE_DIR}" "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  test_dla_step_manager
  revngDataLayoutAnalysis
  revngModel
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_dla_step_manager COMMAND test_dla_step_manager)

#
# test_dla_steps
#

revng_add_test_executable(test_dla_steps "${SRC}/DLASteps.cpp")
target_compile_definitions(test_dla_steps PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_dla_steps PRIVATE "${CMAKE_SOURCE_DIR}"
                                                  "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  test_dla_steps
  revngDataLayoutAnalysis
  revngModel
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_dla_steps COMMAND test_dla_steps)

#
# test_clift
#

revng_add_test_executable(test_clift "${SRC}/Clift.cpp")
target_compile_definitions(test_clift PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_clift PRIVATE "${CMAKE_SOURCE_DIR}"
                                              "${Boost_INCLUDE_DIRS}")
target_link_libraries(test_clift MLIRCliftDialect Boost::unit_test_framework
                      revngUnitTestHelpers ${LLVM_LIBRARIES})
add_test(NAME test_clift COMMAND test_clift)

#
# test_clift_type
#

revng_add_test_executable(test_clift_type "${SRC}/CliftType.cpp")
target_compile_definitions(test_clift_type PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_clift_type PRIVATE "${CMAKE_SOURCE_DIR}"
                                                   "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  test_clift_type
  MLIRCliftDialect
  MLIRCliftUtils
  Boost::unit_test_framework
  revngModel
  revngUnitTestHelpers
  ${LLVM_LIBRARIES})
add_test(NAME test_clift_type COMMAND test_clift_type)

#
# test_pointer_array_emission
#

revng_add_test_executable(test_pointer_array_emission
                          "${SRC}/PointerArrayEmission.cpp")
target_compile_definitions(test_pointer_array_emission
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(
  test_pointer_array_emission PRIVATE "${CMAKE_SOURCE_DIR}"
                                      "${Boost_INCLUDE_DIRS}")
target_link_libraries(
  test_pointer_array_emission
  revngSupport
  revngTypeNames
  revngModel
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
add_test(NAME test_pointer_array_emission COMMAND test_pointer_array_emission)

#
# compile a pass
#

# We use the plain `add_library` and not the revng version
# (revng_add_library_internal) in order to not get this installed into `lib`
add_library(test_scopegraph SHARED "${SRC}/ScopeGraphLoggerPass.cpp")
target_link_libraries(test_scopegraph revngScopeGraphUtils revngSupport
                      ${LLVM_LIBRARIES})

#
# test_statement_comment_placement
#

revng_add_test_executable(test_statement_comment_placement
                          "${SRC}/StatementCommentPlacement.cpp")
target_compile_definitions(test_statement_comment_placement
                           PRIVATE "BOOST_TEST_DYN_LINK=1")
target_include_directories(test_statement_comment_placement
                           PRIVATE "${CMAKE_SOURCE_DIR}")
target_link_libraries(
  test_statement_comment_placement
  revngModel
  revngPTML
  revngSupport
  revngUnitTestHelpers
  Boost::unit_test_framework
  ${LLVM_LIBRARIES})
revng_add_test(NAME test_statement_comment_placement COMMAND
               test_statement_comment_placement)
set_tests_properties(test_statement_comment_placement PROPERTIES LABELS "unit")
