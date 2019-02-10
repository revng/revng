revng_add_analyses_library(RemoveBadPCPass
                           revngc
                           RemoveBadPCPass.cpp)

target_link_libraries(RemoveBadPCPass revng::revngSupport)
