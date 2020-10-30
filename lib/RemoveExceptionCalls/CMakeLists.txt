revng_add_analyses_library(RemoveExceptionCalls
                           revngc
                           RemoveExceptionCallsPass.cpp)

target_link_libraries(RemoveExceptionCalls revng::revngSupport)
