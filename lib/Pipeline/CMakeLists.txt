#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngPipeline
  SHARED
  ContainerSet.cpp
  Context.cpp
  Contract.cpp
  DescriptionConverter.cpp
  Errors.cpp
  GenericLLVMPipe.cpp
  Kind.cpp
  LLVMContainer.cpp
  Loader.cpp
  Pipe.cpp
  Runner.cpp
  RegisterKind.cpp
  Registry.cpp
  Step.cpp
  ExecutionContext.cpp
  Target.cpp
  Global.cpp
  GlobalsMap.cpp)

target_link_libraries(revngPipeline revngStorage revngSupport ${LLVM_LIBRARIES})

set(PIPELINEDESCRIPTION_HEADERS_DIR
    "${CMAKE_SOURCE_DIR}/include/revng/Pipeline/Description")
set(PIPELINEDESCRIPTION_SCHEMA_PATH
    "${CMAKE_SOURCE_DIR}/include/revng/Pipeline/Description/pipelineDescription-schema.yml"
)
set(PIPELINEDESCRIPTION_JSONSCHEMA_PATH
    "${CMAKE_BINARY_DIR}/pipelineDescription-jsonschema.yml")
set(PIPELINEDESCRIPTION_TYPESCRIPT_PATH
    "${CMAKE_BINARY_DIR}/lib/typescript/pipeline-description.ts")
target_tuple_tree_generator(
  revngPipeline
  INSTALL
  HEADER_DIRECTORY
  Pipeline/Description
  NAMESPACE
  pipeline::description
  SCHEMA_PATH
  ${PIPELINEDESCRIPTION_SCHEMA_PATH}
  JSONSCHEMA_PATH
  "${PIPELINEDESCRIPTION_JSONSCHEMA_PATH}"
  PYTHON_PATH
  "${CMAKE_BINARY_DIR}/${PYTHON_INSTALL_PATH}/revng/pipeline_description/_generated.py"
  TYPESCRIPT_PATH
  "${PIPELINEDESCRIPTION_TYPESCRIPT_PATH}"
  ROOT_TYPE
  PipelineDescription
  GLOBAL_NAME
  PipelineDescription)
