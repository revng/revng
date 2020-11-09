#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(DecompilerResourceFinder revngc
  ResourceFinder.cpp)

target_link_libraries(DecompilerResourceFinder
  revng::revngSupport
  ${LLVM_LIBRARIES})
