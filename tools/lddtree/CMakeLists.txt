#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(lddtree Main.cpp)

target_link_libraries(lddtree revngSupport ${LLVM_LIBRARIES})
