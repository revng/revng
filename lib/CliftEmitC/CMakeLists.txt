#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngCliftEmitC
  SHARED
  CBackend.cpp
  CBackendPass.cpp
  CEmitter.cpp
  CSemantics.cpp
  CSemanticsPass.cpp)

target_link_libraries(revngCliftEmitC PUBLIC MLIRPass revngClift revngPTML)

revng_add_analyses_library_internal(revngCliftEmitCAnalyses CBackendPipe.cpp)

target_link_libraries(revngCliftEmitCAnalyses PUBLIC revngCliftEmitC
                                                     revngCliftContainer)
