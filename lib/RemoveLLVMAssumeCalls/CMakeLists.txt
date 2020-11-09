#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(RemoveLLVMAssumeCalls
                           revngc
                           RemoveLLVMAssumeCallsPass.cpp)

target_link_libraries(RemoveLLVMAssumeCalls revng::revngSupport)
