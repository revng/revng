#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngTypeShrinking revngc
  TypeShrinking.cpp
  BitLiveness.cpp
  DataFlowGraph.cpp)

target_link_libraries(revngTypeShrinking
  IsolatedFunctions
  revng::revngModel
  revng::revngSupport)
