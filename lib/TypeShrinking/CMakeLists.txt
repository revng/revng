#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngTypeShrinking
  revngc
  DefUse.cpp)

target_link_libraries(revngTypeShrinking
  revngSupport)
