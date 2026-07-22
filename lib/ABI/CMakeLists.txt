#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngABI
  Analyses/ConvertFunctionsToCABI.cpp
  Analyses/ConvertFunctionsToRaw.cpp
  DefaultFunctionPrototype.cpp
  FunctionType/Conversion.cpp
  FunctionType/Layout.cpp
  FunctionType/ValueDistributor.cpp
  ModelHelpers.cpp
  RegisterStateDeductions.cpp)

target_link_libraries(
  revngABI
  revngModel
  revngModelPasses
  revngPipeline
  revngPipes
  revngSupport
  ${LLVM_LIBRARIES})
