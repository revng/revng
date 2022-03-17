#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcRemoveLLVMDbgIntrinsics revngc
                           RemoveLLVMDbgIntrinsicsPass.cpp)

target_link_libraries(revngcRemoveLLVMDbgIntrinsics revng::revngModel
                      revng::revngSupport ${LLVM_LIBRARIES})
