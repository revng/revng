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

target_link_libraries(revngCliftEmitC MLIRPass revngClift revngPTML)
