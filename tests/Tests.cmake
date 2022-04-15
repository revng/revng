#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

enable_testing()

# Give control to the various subdirectories
include(${CMAKE_SOURCE_DIR}/tests/unit/UnitTests.cmake)
include(${CMAKE_SOURCE_DIR}/tests/analysis/AnalysisTests.cmake)
include(${CMAKE_SOURCE_DIR}/tests/runtime/RuntimeTests.cmake)

add_subdirectory(tests/tools)
add_subdirectory(tests/abi)
add_subdirectory(tests/tuple-tree-generator/python-wrappers/multiple-versions)

set(TEST_CFLAGS_${ARCH} "${TEST_CFLAGS_${ARCH}} -mthumb")
