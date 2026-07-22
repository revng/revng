#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Define revngModel library
revng_add_analyses_library_internal(
  revngModel
  Binary.cpp
  ABIDefinition.cpp
  CommonTypeMethods.cpp
  FunctionTags.cpp
  InlineSegmentGlobalGetters.cpp
  LoadModelPass.cpp
  GlobalVariableBuilder.cpp
  NameBuilder.cpp
  TypeSystemPrinter.cpp
  Processing.cpp
  ProgramCounterHandler.cpp
  RawBinaryView.cpp
  Type.cpp
  TypeDefinition.cpp
  Verification.cpp
  Visits.cpp)

target_link_libraries(revngModel revngSupport)

set(SCHEMA_PATH "${CMAKE_SOURCE_DIR}/include/revng/Model/model-schema.yml")
set(JSONSCHEMA_PATH "${CMAKE_BINARY_DIR}/model-jsonschema.yml")
set(TYPESCRIPT_PATH "${CMAKE_BINARY_DIR}/lib/typescript/model.ts")
set(DOCS_PATH "${CMAKE_BINARY_DIR}/share/doc/revng/references/model.md")
target_tuple_tree_generator(
  revngModel
  INSTALL
  HEADER_DIRECTORY
  Model
  NAMESPACE
  model
  SCHEMA_PATH
  "${SCHEMA_PATH}"
  JSONSCHEMA_PATH
  "${JSONSCHEMA_PATH}"
  PYTHON_PATH
  "${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}/${PYTHON_GENERATED_MODEL_PATH}"
  TYPESCRIPT_PATH
  "${TYPESCRIPT_PATH}"
  TYPESCRIPT_INCLUDE
  "${CMAKE_SOURCE_DIR}/typescript/model.ts"
  DOCS_PATH
  "${DOCS_PATH}"
  EMIT_TRACKING
  GLOBAL_NAME
  Model
  PYTHON_MIXINS
  AllMixin
  BinaryMixin
  FunctionMixin
  TypeDefinitionMixin
  SEPARATE_STRING_TYPES
  "MetaAddress"
  SCALAR_TYPES
  "MetaAddress")

# TODO: merge into the main model generator above.
target_tuple_tree_generator(
  revngModel
  INSTALL
  HEADER_DIRECTORY
  "Model/ABI"
  NAMESPACE
  "abi"
  SCHEMA_PATH
  "${CMAKE_SOURCE_DIR}/include/revng/Model/ABI/schema.yml"
  SCALAR_TYPES
  "model::ABI::Values"
  "model::PrimitiveKind::Values"
  "model::Register::Values")

add_subdirectory(Pass)
add_subdirectory(Importer)
