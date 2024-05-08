#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

#
# Pipeline
#

# Define StringContainerLibrary
revng_add_library_internal(
  revngStringContainerLibrary SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/StringContainerLibrary.cpp)
target_link_libraries(revngStringContainerLibrary revngPipeline revngPipes)

# Define BadBehaviorLibrary
revng_add_library_internal(revngBadBehaviorLibrary SHARED
                           ${CMAKE_CURRENT_SOURCE_DIR}/BadBehaviorLibrary.cpp)
target_link_libraries(revngBadBehaviorLibrary revngSupport)

macro(
  add_pipeline_test
  TEST_NAME
  PIPELINE_FILE
  TARGETS
  INPUTS_LIST
  OUTPUTS_LIST
  FLAGS)
  revng_add_test(
    NAME
    "${TEST_NAME}"
    COMMAND
    bash
    -c
    "rm -rf ${CMAKE_BINARY_DIR}/${TEST_NAME} &&
      ${CMAKE_BINARY_DIR}/libexec/revng/revng-pipeline \
      -P ${CMAKE_CURRENT_SOURCE_DIR}/${PIPELINE_FILE} \
      ${TARGETS} \
      -i ${INPUTS_LIST} \
      -o ${OUTPUTS_LIST} \
      -l ${CMAKE_BINARY_DIR}/lib/librevngStringContainerLibrary.so \
      --resume ${CMAKE_BINARY_DIR}/${TEST_NAME} \
      -f ${FLAGS}")
  set_tests_properties(
    "${TEST_NAME}" PROPERTIES LABELS "pipeline;run" ENVIRONMENT
                              "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}")
endmacro()

macro(add_pipeline_dump_test TEST_NAME PIPELINE_FILE)
  revng_add_test(
    NAME
    "${TEST_NAME}"
    COMMAND
    "${CMAKE_BINARY_DIR}/libexec/revng/revng-pipeline"
    -P
    "${CMAKE_CURRENT_SOURCE_DIR}/${PIPELINE_FILE}"
    --produce=first-step:dc:dc:string-kind
    -l
    "${CMAKE_BINARY_DIR}/lib/librevngStringContainerLibrary.so"
    -d)
  set_tests_properties(
    "${TEST_NAME}" PROPERTIES LABELS "pipeline;dump" ENVIRONMENT
                              "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}")
endmacro()

macro(
  test_pipeline_output
  TEST_NAME
  PIPELINE_FILE
  TARGETS
  INPUTS_LIST
  OUTPUTS_LIST
  FLAGS
  FILE_TO_CHECK
  EXPECTED_OUTPUT)

  add_pipeline_test("${TEST_NAME}" "${PIPELINE_FILE}" "${TARGETS}"
                    "${INPUTS_LIST}" "${OUTPUTS_LIST}" "${FLAGS}")
  revng_add_test(
    NAME
    "${TEST_NAME}-check"
    COMMAND
    diff
    -u
    "${FILE_TO_CHECK}"
    "${EXPECTED_OUTPUT}")

  set_tests_properties("${TEST_NAME}-check" PROPERTIES DEPENDS "${TEST_NAME}"
                                                       LABELS "pipeline;check")
endmacro()

macro(
  ensure_pipeline_failure
  TEST_NAME
  PIPELINE_FILE
  TARGETS
  INPUTS_LIST
  OUTPUTS_LIST
  FLAGS)
  add_pipeline_test(
    "${TEST_NAME}"
    "${PIPELINE_FILE}"
    "${TARGETS}"
    "${INPUTS_LIST}"
    "${OUTPUTS_LIST}"
    "${FLAGS}"
    "${EXPECTED_OUTPUT}")
  set_tests_properties("${TEST_NAME}" PROPERTIES WILL_FAIL TRUE)
endmacro()

set(COPY_PIPE_TEST_INPUTS
    "${CMAKE_CURRENT_SOURCE_DIR}/CopyPipeTestInput.txt:begin/strings-1")
set(COPY_PIPE_TEST_OUTPUTS "CopyPipeTestOutput.txt:first-step/strings-2")

test_pipeline_output(
  pipeline-copy-pipe
  CopyPipeTestPipeline.yml
  --produce=first-step/strings-2/root:string-kind
  "${COPY_PIPE_TEST_INPUTS}"
  "${COPY_PIPE_TEST_OUTPUTS}"
  None
  CopyPipeTestOutput.txt
  "${CMAKE_CURRENT_SOURCE_DIR}/CopyPipeTestExpected.txt")

set(ROLE_COPY_PIPE_TEST_OUTPUTS
    "RoleCopyPipeTestOutput.txt:first-step/strings-2")

test_pipeline_output(
  role-pipeline-copy-pipe
  RoleCopyPipeTestPipeline.yml
  --produce=first-step/strings-2/root:string-kind
  "${COPY_PIPE_TEST_INPUTS}"
  "${ROLE_COPY_PIPE_TEST_OUTPUTS}"
  None
  RoleCopyPipeTestOutput.txt
  "${CMAKE_CURRENT_SOURCE_DIR}/CopyPipeTestExpected.txt")

# Try to produce a target produced by a non-disabled pipe
test_pipeline_output(
  pipeline-copy-pipe-flag
  CopyPipeFlagTestPipeline.yml
  --produce=first-step/strings-2/root:string-kind
  "${COPY_PIPE_TEST_INPUTS}"
  "${COPY_PIPE_TEST_OUTPUTS}"
  None
  CopyPipeTestOutput.txt
  "${CMAKE_CURRENT_SOURCE_DIR}/CopyPipeTestExpected.txt")

# Try to produce a target produced by a disabled pipe
ensure_pipeline_failure(
  pipeline-flag-fail-test CopyPipeFlagTestPipeline.yml
  --produce=first-step/strings-2/root:string-kind "${COPY_PIPE_TEST_INPUTS}"
  "${COPY_PIPE_TEST_OUTPUTS}" disable-copy-pipe)

# Try to invoke a non-existing pass using a PureLLVMPipe
ensure_pipeline_failure(
  pipeline-test-missing-pipe MissingPassPipelineTest.yml
  --produce=first-step/strings-2/root:string-kind "${COPY_PIPE_TEST_INPUTS}"
  "${COPY_PIPE_TEST_OUTPUTS}" dont-care)

# Test an LLVMPass pipe
add_pipeline_dump_test(pipeline-llvm-pass-test PassPipelineTest.yml)

# Test running revng-pipeline in multiple invocation
revng_add_test(
  NAME
  pipeline-multistep-test
  COMMAND
  "${CMAKE_CURRENT_SOURCE_DIR}/MultiStepPipelineTest.sh"
  --load
  ${CMAKE_BINARY_DIR}/lib/librevngStringContainerLibrary.so
  WORKING_DIRECTORY
  "${CMAKE_BINARY_DIR}")
set_tests_properties(pipeline-multistep-test PROPERTIES DEPENDS revng-pipeline
                                                        LABELS "pipeline")

#
# Invalidate
#

function(add_invalidation_test TEST_NAME)
  set(options)
  set(oneValueArgs STEP INPUT PIPELINE INVALIDATIONS)
  set(multiValueArgs TARGETS)
  cmake_parse_arguments(OPT "${options}" "${oneValueArgs}" "${multiValueArgs}"
                        "${ARGN}")

  revng_add_test(
    NAME
    "${TEST_NAME}"
    COMMAND
    bash
    -c
    "rm -rf ${CMAKE_BINARY_DIR}/${TEST_NAME} &&
      mkdir ${CMAKE_BINARY_DIR}/${TEST_NAME} &&
      ${CMAKE_BINARY_DIR}/libexec/revng/revng-pipeline \
      -P ${OPT_PIPELINE} \
      ${OPT_TARGETS} \
      -i ${OPT_INPUT}:begin/strings-1 \
      -l ${CMAKE_BINARY_DIR}/lib/librevngStringContainerLibrary.so \
      --resume ${CMAKE_BINARY_DIR}/${TEST_NAME} &&
      ${CMAKE_BINARY_DIR}/libexec/revng/revng-invalidate \
      -P ${OPT_PIPELINE} \
      ${OPT_INVALIDATIONS} \
      -l ${CMAKE_BINARY_DIR}/lib/librevngStringContainerLibrary.so \
      --resume ${CMAKE_BINARY_DIR}/${TEST_NAME}")
  set_tests_properties(
    "${TEST_NAME}" PROPERTIES LABELS "pipeline;run" ENVIRONMENT
                              "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}")
endfunction()

add_invalidation_test(
  simple_invalidation_test
  TARGETS
  "--produce=first-step/strings-2/root:string-kind"
  STEP
  "first-step"
  INPUT
  "${CMAKE_CURRENT_SOURCE_DIR}/InvalidateTestInput.txt"
  PIPELINE
  "${CMAKE_CURRENT_SOURCE_DIR}/SimpleInvalidationTestPipeline.yml"
  INVALIDATIONS
  "first-step/strings-1/root:string-kind")

#
# Bad behavior handling
#

macro(add_crash_test TEST_NAME SIGNAL EXPECTED_STATUS)
  revng_add_test(
    NAME
    "${TEST_NAME}"
    COMMAND
    bash
    ${CMAKE_CURRENT_SOURCE_DIR}/run_crash_test.sh
    ${CMAKE_BINARY_DIR}
    ${EXPECTED_STATUS}
    ${SIGNAL})
  set_tests_properties(
    "${TEST_NAME}" PROPERTIES LABELS "pipeline;crash" ENVIRONMENT
                              "PATH=${CMAKE_BINARY_DIR}/bin:$ENV{PATH}")
endmacro()

add_crash_test(crash_test_sigill 4 132)

add_crash_test(crash_test_sigabrt 6 134)

add_crash_test(crash_test_sigsegv 11 139)

# The following end up segfaulting in the signal handler due to unwinding issues
add_crash_test(crash_test_sigquit 3 134)

add_crash_test(crash_test_sigtrap 5 134)

add_crash_test(crash_test_sigbus 7 134)

add_crash_test(crash_test_sigfpe 8 134)
