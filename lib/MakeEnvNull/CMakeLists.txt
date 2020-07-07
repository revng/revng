#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(MakeEnvNull revngc MakeEnvNull.cpp)

target_link_libraries(MakeEnvNull revng::revngSupport ${LLVM_LIBRARIES})


