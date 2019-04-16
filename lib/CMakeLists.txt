#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

#
# Handle the analysis libraries
#
make_directory("${CMAKE_BINARY_DIR}/analyses/")

add_custom_target(analyses)

macro(revng_add_analyses_library_internal NAME)
  revng_add_analyses_library("${NAME}" revng ${ARGN})
  add_custom_command(TARGET "${NAME}" POST_BUILD VERBATIM
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${NAME}>" "${CMAKE_BINARY_DIR}/analyses/")
  add_dependencies(analyses "${NAME}")
endmacro()

add_subdirectory(BasicAnalyses)
add_subdirectory(Dump)
add_subdirectory(FunctionCallIdentification)
add_subdirectory(FunctionIsolation)
add_subdirectory(ReachingDefinitions)
add_subdirectory(StackAnalysis)
add_subdirectory(Support)
add_subdirectory(UnitTestHelpers)
