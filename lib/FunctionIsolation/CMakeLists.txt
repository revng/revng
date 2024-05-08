#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

revng_add_analyses_library_internal(
  revngFunctionIsolation
  DropRoot.cpp
  EnforceABI.cpp
  InlineHelpers.cpp
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
  revngSupport
  revngPipes
  ${LLVM_LIBRARIES})
