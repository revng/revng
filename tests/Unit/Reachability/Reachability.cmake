#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

set(SRC_LIB "${SRC}/Reachability")

add_library(revngcReachability SHARED "${SRC_LIB}/ReachabilityPass.cpp")
target_link_libraries(revngcReachability
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})
