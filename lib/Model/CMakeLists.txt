#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Define revngModel library
revng_add_analyses_library_internal(
  revngModel
  Binary.cpp
  LoadModelPass.cpp
  TypeSystemPrinter.cpp
  Processing.cpp
  SerializeModelPass.cpp
  Type.cpp
  Visits.cpp)

target_link_libraries(revngModel revngSupport)

# Header files containing model YAML must be added here
set(MODEL_HEADERS_DIR "${CMAKE_SOURCE_DIR}/include/revng/Model")
set(SCHEMA_PATH "${CMAKE_BINARY_DIR}/model-schema.yml")
set(JSONSCHEMA_PATH "${CMAKE_BINARY_DIR}/model-jsonschema.yml")
set(TYPESCRIPT_PATH "${CMAKE_BINARY_DIR}/lib/typescript/model.ts")
target_tuple_tree_generator(
  revngModel
  INSTALL
  HEADER_DIRECTORY
  Model
  NAMESPACE
  model
  SCHEMA_PATH
  ${SCHEMA_PATH}
  JSONSCHEMA_PATH
  "${JSONSCHEMA_PATH}"
  PYTHON_PATH
  "${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}/${PYTHON_GENERATED_MODEL_PATH}"
  TYPESCRIPT_PATH
  "${TYPESCRIPT_PATH}"
  TYPESCRIPT_INCLUDE
  "${CMAKE_SOURCE_DIR}/typescript/model.ts"
  ROOT_TYPE
  Binary
  EMIT_TRACKING
  GLOBAL_NAME
  Model
  STRING_TYPES
  "Identifier"
  SEPARATE_STRING_TYPES
  "MetaAddress"
  SCALAR_TYPES
  "Identifier"
  "MetaAddress"
  HEADERS
  "${MODEL_HEADERS_DIR}/ABI.h"
  "${MODEL_HEADERS_DIR}/Architecture.h"
  "${MODEL_HEADERS_DIR}/Argument.h"
  "${MODEL_HEADERS_DIR}/Binary.h"
  "${MODEL_HEADERS_DIR}/CABIFunctionType.h"
  "${MODEL_HEADERS_DIR}/CanonicalRegisterValue.h"
  "${MODEL_HEADERS_DIR}/CallSitePrototype.h"
  "${MODEL_HEADERS_DIR}/DynamicFunction.h"
  "${MODEL_HEADERS_DIR}/EnumEntry.h"
  "${MODEL_HEADERS_DIR}/EnumType.h"
  "${MODEL_HEADERS_DIR}/Function.h"
  "${MODEL_HEADERS_DIR}/FunctionAttribute.h"
  "${MODEL_HEADERS_DIR}/NamedTypedRegister.h"
  "${MODEL_HEADERS_DIR}/PrimitiveType.h"
  "${MODEL_HEADERS_DIR}/PrimitiveTypeKind.h"
  "${MODEL_HEADERS_DIR}/QualifiedType.h"
  "${MODEL_HEADERS_DIR}/Qualifier.h"
  "${MODEL_HEADERS_DIR}/QualifierKind.h"
  "${MODEL_HEADERS_DIR}/RawFunctionType.h"
  "${MODEL_HEADERS_DIR}/Relocation.h"
  "${MODEL_HEADERS_DIR}/RelocationType.h"
  "${MODEL_HEADERS_DIR}/Register.h"
  "${MODEL_HEADERS_DIR}/Section.h"
  "${MODEL_HEADERS_DIR}/Segment.h"
  "${MODEL_HEADERS_DIR}/StructField.h"
  "${MODEL_HEADERS_DIR}/StructType.h"
  "${MODEL_HEADERS_DIR}/Type.h"
  "${MODEL_HEADERS_DIR}/TypedRegister.h"
  "${MODEL_HEADERS_DIR}/TypedefType.h"
  "${MODEL_HEADERS_DIR}/UnionField.h"
  "${MODEL_HEADERS_DIR}/UnionType.h")

add_subdirectory(Pass)
add_subdirectory(Importer)
