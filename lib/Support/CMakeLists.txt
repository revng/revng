#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcSupport
  revngc
  FunctionTags.cpp)

target_link_libraries(revngcSupport
  revng::revngSupport)
