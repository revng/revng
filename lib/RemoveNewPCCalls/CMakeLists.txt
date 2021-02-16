#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(RemoveNewPCCalls
                           revngc
                           RemoveNewPCCallsPass.cpp)

target_link_libraries(RemoveNewPCCalls
  IsolatedFunctions
  revng::revngModel
  revng::revngSupport)
