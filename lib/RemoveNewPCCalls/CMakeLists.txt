revng_add_analyses_library(RemoveNewPCCalls
                           revngc
                           RemoveNewPCCallsPass.cpp)

target_link_libraries(RemoveNewPCCalls revng::revngSupport)
