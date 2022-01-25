set(TEST_DIR "${CMAKE_BINARY_DIR}/tests/python/multiversion")
set(MODULE_DIR "${CMAKE_BINARY_DIR}/tests/python/multiversion/testmodule")

file(MAKE_DIRECTORY "${TEST_DIR}")
file(MAKE_DIRECTORY "${MODULE_DIR}/v1")
file(MAKE_DIRECTORY "${MODULE_DIR}/v2")

set(JSONSCHEMA_V1 "${CMAKE_CURRENT_SOURCE_DIR}/v1_schema.yaml")
set(JSONSCHEMA_V2 "${CMAKE_CURRENT_SOURCE_DIR}/v2_schema.yaml")
set(GENERATED_V1 "${MODULE_DIR}/v1/_generated.py")
set(GENERATED_V2 "${MODULE_DIR}/v2/_generated.py")

# Copy the model._common directory in the build directory
file(COPY
  "${CMAKE_SOURCE_DIR}/python/revng/model/_common"
  DESTINATION
  "${MODULE_DIR}"
)

# Copy the base class
file(COPY "base.py" DESTINATION "${MODULE_DIR}/v1")
file(COPY "base.py" DESTINATION "${MODULE_DIR}/v2")

# Generate python model for v1 and v2
add_custom_command(
  OUTPUT "${GENERATED_V1}"
  COMMAND "datamodel-codegen"
  ARGS
  --base-class .base.MonkeyPatchingBaseClass
  --target-python-version 3.6
  --input "${JSONSCHEMA_V1}"
  > "${GENERATED_V1}"
)

add_custom_command(
  OUTPUT "${GENERATED_V2}"
  COMMAND "datamodel-codegen"
  ARGS
  --base-class .base.MonkeyPatchingBaseClass
  --target-python-version 3.6
  --input "${JSONSCHEMA_V2}"
  > "${GENERATED_V2}"
)

add_custom_target(
  test_deserialize_multiple_versions_generated_files
  ALL
  DEPENDS "${GENERATED_V1}" "${GENERATED_V2}"
)

add_test(
  NAME test_deserialize_multiple_versions
  COMMAND env "PYTHONPATH=${TEST_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/deserialize_multiple_versions.py"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
set_tests_properties(test_deserialize_multiple_versions PROPERTIES LABELS "unit")
