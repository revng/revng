#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngFunctionIsolation
  EnforceABI.cpp
  InlineHelpers.cpp
  InvokeIsolatedFunctions.cpp
  IsolateFunctions.cpp
  PromoteCSVs.cpp
  RemoveExceptionalCalls.cpp
  StructInitializers.cpp)

target_link_libraries(revngFunctionIsolation
  revngModel
  revngStackAnalysis
  revngSupport)
