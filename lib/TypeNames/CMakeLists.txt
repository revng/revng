#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcTypeNames revngc LLVMTypeNames.cpp
                           ModelTypeNames.cpp ModelToPTMLTypeHelpers.cpp)

target_link_libraries(revngcTypeNames revng::revngABI revng::revngModel
                      revng::revngSupport revngcSupport ${LLVM_LIBRARIES})
