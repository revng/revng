revng_add_analyses_library(RemoveLLVMAssumeCalls
                           revngc
                           RemoveLLVMAssumeCallsPass.cpp)

target_link_libraries(RemoveLLVMAssumeCalls revng::revngSupport)
