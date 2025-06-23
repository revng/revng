#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Define revngModel library
revng_add_analyses_library_internal(
  revngModel
  Binary.cpp
  CommonTypeMethods.cpp
  LoadModelPass.cpp
  NameBuilder.cpp
  TypeSystemPrinter.cpp
  Processing.cpp
  Type.cpp
  TypeDefinition.cpp
  Verification.cpp
  Visits.cpp)

target_link_libraries(revngModel revngSupport)

# Header files containing model YAML must be added here
set(MODEL_HEADERS_DIR "${CMAKE_SOURCE_DIR}/include/revng/Model")
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
  "${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}/revng/model/mixins.py"
  SEPARATE_STRING_TYPES
  "MetaAddress"
  SCALAR_TYPES
  "MetaAddress")

add_subdirectory(Pass)
add_subdirectory(Importer)
