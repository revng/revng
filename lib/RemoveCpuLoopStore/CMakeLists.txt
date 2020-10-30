revng_add_analyses_library(RemoveCpuLoopStore
                           revngc
                           RemoveCpuLoopStorePass.cpp)

target_link_libraries(RemoveCpuLoopStore revng::revngSupport)
