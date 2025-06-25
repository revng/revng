#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngABI
  Analyses/ConvertFunctionsToCABI.cpp
  Analyses/ConvertFunctionsToRaw.cpp
  DefaultFunctionPrototype.cpp
  Definition.cpp
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

set(HEADERS_REQUIRING_TTG "${CMAKE_SOURCE_DIR}/include/revng/ABI")
target_tuple_tree_generator(
  revngABI
  INSTALL
  HEADER_DIRECTORY
  "ABI"
  NAMESPACE
  "abi"
  SCHEMA_PATH
  "${CMAKE_SOURCE_DIR}/include/revng/ABI/abi-internal-schema.yml"
  SCALAR_TYPES
  "model::ABI::Values"
  "model::Register::Values")
