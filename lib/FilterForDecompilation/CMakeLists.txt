#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcFilterForDecompilation revngc
                           FilterForDecompilationPass.cpp)

target_link_libraries(revngcFilterForDecompilation revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES})
