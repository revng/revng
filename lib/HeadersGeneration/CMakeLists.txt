#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

# ModelToHeader

revng_add_analyses_library(
  revngcModelToHeader
  revngc
  ModelToHeader.cpp
  ModelToHeaderPipe.cpp
  DependencyGraph.cpp
  ModelTypeDefinition.cpp
  ModelTypeDefinitionPipe.cpp)

target_link_libraries(
  revngcModelToHeader
  revngcTypeNames
  revng::revngModel
  revng::revngSupport
  revng::revngPipeline
  revng::revngPipes
  revng::revngPTML
  ${LLVM_LIBRARIES})

# HelpersToHeader

revng_add_analyses_library(revngcHelpersToHeader revngc HelpersToHeader.cpp
                           HelpersToHeaderPipe.cpp)

target_link_libraries(
  revngcHelpersToHeader
  revngcTypeNames
  revngcSupport
  revng::revngModel
  revng::revngSupport
  revng::revngPipeline
  revng::revngPipes
  revng::revngPTML
  ${LLVM_LIBRARIES})
