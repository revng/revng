#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(RemoveExceptionCalls
  revngc
  RemoveExceptionCallsPass.cpp)

target_link_libraries(RemoveExceptionCalls
  IsolatedFunctions
  revng::revngModel
  revng::revngSupport)
