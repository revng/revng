#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

set(SRC_LIB "${SRC}/Reachability")

add_library(Reachability SHARED "${SRC_LIB}/ReachabilityPass.cpp")
target_link_libraries(Reachability ${LLVM_LIBRARIES})
