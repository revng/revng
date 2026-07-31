#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngFunctionIsolation
  EnforceABI.cpp
  InlineAlwaysInlineFunctions.cpp
  InvokeIsolatedFunctions.cpp
  IsolateFunctions.cpp
  PromoteCSVs.cpp
  RemoveExceptionalCalls.cpp
  StripDebugInfoFromHelpers.cpp
  StructInitializers.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES TransformUtils)

target_link_libraries(
  revngFunctionIsolation
  revngABI
  revngModel
  revngEarlyFunctionAnalysis
  revngInlineHelpers
  revngSupport
  ${LLVM_LIBRARIES})
