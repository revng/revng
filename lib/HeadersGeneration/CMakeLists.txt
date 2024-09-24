#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# ModelToHeader

revng_add_analyses_library(
  revngcModelToHeader revngc ModelToHeader.cpp ModelToHeaderPipe.cpp
  ModelTypeDefinitionPipe.cpp Options.cpp)

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
