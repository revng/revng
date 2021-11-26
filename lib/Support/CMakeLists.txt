#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(Support
  revngc
  FunctionTags.cpp)

target_link_libraries(Support
  revng::revngSupport)
