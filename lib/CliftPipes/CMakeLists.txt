#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngCliftPipes
  CliftContainer.cpp
  Clifter.cpp
  EmitC.cpp
  ImportDescriptiveInfo.cpp
  ImportTypes.cpp
  ModelVerifyClift.cpp
  Optimization.cpp)

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
