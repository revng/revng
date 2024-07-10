#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngTypeShrinking EarlyTypeShrinking.cpp TypeShrinking.cpp BitLiveness.cpp
  DataFlowGraph.cpp)

target_link_libraries(revngTypeShrinking revngSupport ${LLVM_LIBRARIES})
