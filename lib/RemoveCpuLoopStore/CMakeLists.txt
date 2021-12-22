#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcRemoveCpuLoopStore
  revngc
  RemoveCpuLoopStorePass.cpp)

target_link_libraries(revngcRemoveCpuLoopStore
  revng::revngModel
  revng::revngSupport)
