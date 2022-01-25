#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# Header files containing model YAML must be added here
set(MODEL_HEADERS_DIR "${CMAKE_SOURCE_DIR}/include/revng/Model")
set(MODEL_HEADERS
  "${MODEL_HEADERS_DIR}/CABIFunctionType.h"
  "${MODEL_HEADERS_DIR}/EnumType.h"
  "${MODEL_HEADERS_DIR}/PrimitiveType.h"
  "${MODEL_HEADERS_DIR}/PrimitiveTypeKind.h"
  "${MODEL_HEADERS_DIR}/RawFunctionType.h"
  "${MODEL_HEADERS_DIR}/StructType.h"
  "${MODEL_HEADERS_DIR}/TypedefType.h"
  "${MODEL_HEADERS_DIR}/Type.h"
  "${MODEL_HEADERS_DIR}/TypeKind.h"
  "${MODEL_HEADERS_DIR}/UnionType.h"
  "${MODEL_HEADERS_DIR}/ABI.h"
  "${MODEL_HEADERS_DIR}/FunctionEdgeBase.h"
  "${MODEL_HEADERS_DIR}/Architecture.h"
  "${MODEL_HEADERS_DIR}/Argument.h"
  "${MODEL_HEADERS_DIR}/BasicBlock.h"
  "${MODEL_HEADERS_DIR}/Binary.h"
  "${MODEL_HEADERS_DIR}/CallEdge.h"
  "${MODEL_HEADERS_DIR}/DynamicFunction.h"
  "${MODEL_HEADERS_DIR}/EnumEntry.h"
  "${MODEL_HEADERS_DIR}/FunctionEdge.h"
  "${MODEL_HEADERS_DIR}/FunctionEdgeType.h"
  "${MODEL_HEADERS_DIR}/Function.h"
  "${MODEL_HEADERS_DIR}/FunctionAttribute.h"
  "${MODEL_HEADERS_DIR}/FunctionType.h"
  "${MODEL_HEADERS_DIR}/NamedTypedRegister.h"
  "${MODEL_HEADERS_DIR}/QualifiedType.h"
  "${MODEL_HEADERS_DIR}/Qualifier.h"
  "${MODEL_HEADERS_DIR}/QualifierKind.h"
  "${MODEL_HEADERS_DIR}/Register.h"
  "${MODEL_HEADERS_DIR}/RegisterState.h"
  "${MODEL_HEADERS_DIR}/Segment.h"
  "${MODEL_HEADERS_DIR}/StructField.h"
  "${MODEL_HEADERS_DIR}/TypedRegister.h"
  "${MODEL_HEADERS_DIR}/UnionField.h"
)

# Generate C++ headers from the collected YAML
set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/include/revng/Model/Generated")
tuple_tree_generator_cpp_from_headers(
  generate-model-tuple-tree-code
  "${MODEL_HEADERS}"
  TUPLE-TREE-YAML
  model
  "${OUTPUT_DIR}"
  "revng/Model"
  GENERATED_HEADERS
  GENERATED_IMPLS
)

install(
    DIRECTORY ${OUTPUT_DIR}
    DESTINATION include/revng/Model
)

# Define revngModel library
revng_add_analyses_library_internal(
  revngModel
  Binary.cpp
  LoadModelPass.cpp
  Processing.cpp
  SerializeModelPass.cpp
  Type.cpp
  TypesDeduplication.cpp
  ${GENERATED_IMPLS}
)

target_link_libraries(revngModel revngSupport)
add_dependencies(revngModel generate-model-tuple-tree-code)
