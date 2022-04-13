#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcMakeEnvNull revngc MakeEnvNull.cpp)

target_link_libraries(revngcMakeEnvNull revng::revngModel revng::revngSupport
                      ${LLVM_LIBRARIES})
