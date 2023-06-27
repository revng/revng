#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcRemoveExtractValues revngc
                           RemoveExtractValuesPass.cpp)

target_link_libraries(revngcRemoveExtractValues revngcSupport
                      revng::revngSupport ${LLVM_LIBRARIES})
