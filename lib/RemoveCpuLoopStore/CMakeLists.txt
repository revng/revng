#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(RemoveCpuLoopStore
  revngc
  RemoveCpuLoopStorePass.cpp)

target_link_libraries(RemoveCpuLoopStore
  revng::revngModel
  revng::revngSupport)
