#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngFunctionIsolation
  EnforceABI.cpp
  IsolateFunctions.cpp)

target_link_libraries(revngFunctionIsolation
  revngModel
  revngStackAnalysis
  revngSupport)
