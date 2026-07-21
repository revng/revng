#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngCliftPipes
  Clifter.cpp
  EmitC.cpp
  EmitCAsDirectory.cpp
  EmitCAsSingleFile.cpp
  Headers.cpp
  ImportDataModel.cpp
  ImportDescriptiveInfo.cpp
  ImportTypes.cpp
  VerifyAgainstModel.cpp)

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
