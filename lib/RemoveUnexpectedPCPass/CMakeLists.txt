revng_add_analyses_library(RemoveUnexpectedPCPass
                           revngc
                           RemoveUnexpectedPCPass.cpp)

target_link_libraries(RemoveUnexpectedPCPass revng::revngSupport)
