#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcRemoveNewPCCalls revngc
                           RemoveNewPCCallsPass.cpp)

target_link_libraries(revngcRemoveNewPCCalls revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES})
