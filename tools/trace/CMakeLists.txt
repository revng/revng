#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(revng-trace-run run/Main.cpp)
revng_add_executable(revng-trace-inspect inspect/Main.cpp)

target_link_libraries(revng-trace-run revngPipelineC revngSupport
                      ${LLVM_LIBRARIES})
target_link_libraries(revng-trace-inspect revngPipelineC revngSupport
                      ${LLVM_LIBRARIES})
