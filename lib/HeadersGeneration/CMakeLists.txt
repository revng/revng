#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# ModelToHeader

revng_add_analyses_library_internal(
  revngModelToHeader ModelToHeader.cpp ModelToHeaderPipe.cpp
  ModelTypeDefinitionPipe.cpp Options.cpp)

target_link_libraries(
  revngModelToHeader
  revngTypeNames
  revngEarlyFunctionAnalysis
  revngModel
  revngSupport
  revngPipeline
  revngPipes
  revngPTML
  ${LLVM_LIBRARIES})

# HelpersToHeader

revng_add_analyses_library_internal(revngHelpersToHeader HelpersToHeader.cpp
                                    HelpersToHeaderPipe.cpp)

target_link_libraries(
  revngHelpersToHeader
  revngTypeNames
  revngSupport
  revngModel
  revngPipeline
  revngPipes
  revngPTML
  ${LLVM_LIBRARIES})
