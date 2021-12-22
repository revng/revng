#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcValueManipulationAnalysis revngc
  ValueManipulationAnalysis.cpp
  TypeFlowGraph.cpp
  TypeFlowNode.cpp
  Mincut.cpp
  ContractedGraph.cpp)

target_link_libraries(revngcValueManipulationAnalysis
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})
