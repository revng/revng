#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(trace-run run/Main.cpp)
revng_add_executable(trace-inspect inspect/Main.cpp)

target_link_libraries(trace-run revngPipelineC revngSupport ${LLVM_LIBRARIES})
target_link_libraries(trace-inspect revngPipelineC revngSupport
                      ${LLVM_LIBRARIES})
