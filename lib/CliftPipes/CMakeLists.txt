#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngCliftPipes
  CBackendPipe.cpp
  CliftContainer.cpp
  ClifterPipe.cpp
  Headers.cpp
  ImportCliftTypesPipe.cpp
  ImportModelNamesPipe.cpp
  LegalizationPipe.cpp
  ModelVerifyPipe.cpp
  OptimizationPipe.cpp)

target_link_libraries(
  revngCliftPipes
  revngClift
  revngCliftEmitC
  revngClifter
  revngCliftImportModel
  revngCliftTransforms
  revngPipeline
  revngPipes
  MLIRTransforms)
