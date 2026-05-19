#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(merge-llvm Main.cpp)

target_link_libraries(merge-llvm revngSupport ${LLVM_LIBRARIES})
