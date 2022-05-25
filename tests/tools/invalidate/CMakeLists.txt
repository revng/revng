function(add_invalidation_test TEST_NAME)
  set(options)
  set(oneValueArgs STEP INPUT PIPELINE INVALIDATIONS)
  set(multiValueArgs TARGETS)
  cmake_parse_arguments(OPT "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        "${ARGN}")

  add_test(
    NAME "${TEST_NAME}"
    COMMAND
      bash -c "rm -rf ${CMAKE_BINARY_DIR}/${TEST_NAME} &&
      mkdir ${CMAKE_BINARY_DIR}/${TEST_NAME} &&
      ${CMAKE_BINARY_DIR}/libexec/revng/revng-pipeline \
      -P ${OPT_PIPELINE} \
      ${OPT_TARGETS} \
      -i begin:Strings1:${OPT_INPUT} \
      -l ${CMAKE_BINARY_DIR}/lib/librevngStringContainerLibrary.so \
      -p ${CMAKE_BINARY_DIR}/${TEST_NAME} &&
      ${CMAKE_BINARY_DIR}/libexec/revng/revng-invalidate \
      -P ${OPT_PIPELINE} \
      ${OPT_INVALIDATIONS} \
      -l ${CMAKE_BINARY_DIR}/lib/librevngStringContainerLibrary.so \
      -p ${CMAKE_BINARY_DIR}/${TEST_NAME} \
      --step ${OPT_STEP}")
  set_tests_properties(
    "${TEST_NAME}" PROPERTIES LABELS "pipeline;run" ENVIRONMENT
                              "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}")
endfunction()

add_invalidation_test(
  simple_invalidation_test
  TARGETS
  "FirstStep:Strings2:Root:StringKind"
  STEP
  "FirstStep"
  INPUT
  "${CMAKE_CURRENT_SOURCE_DIR}/InvalidateTestInput.txt"
  PIPELINE
  "${CMAKE_CURRENT_SOURCE_DIR}/SimpleInvalidationTestPipeline.yml"
  INVALIDATIONS
  "FirstStep:Strings1:Root:StringKind")
