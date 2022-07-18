#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcValueManipulationAnalysis
  revngc
  ValueManipulationAnalysis.cpp
  TypeFlowGraph.cpp
  TypeFlowNode.cpp
  Mincut.cpp
  ContractedGraph.cpp
  VMAPipeline.cpp)

target_link_libraries(revngcValueManipulationAnalysis revngcSupport
                      revng::revngModel revng::revngSupport ${LLVM_LIBRARIES})
