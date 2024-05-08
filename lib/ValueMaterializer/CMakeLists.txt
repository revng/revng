#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

revng_add_library_internal(
  revngValueMaterializer SHARED AdvancedValueInfo.cpp ControlFlowEdgesGraph.cpp
  DataFlowGraph.cpp ValueMaterializer.cpp)

target_link_libraries(revngValueMaterializer
                      PUBLIC revngSupport revngBasicAnalyses ${LLVM_LIBRARIES})
