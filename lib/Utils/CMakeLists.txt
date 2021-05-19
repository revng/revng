#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(Utils revngc Utils.cpp)

target_link_libraries(Utils revng::revngSupport ${LLVM_LIBRARIES})
