revng_add_analyses_library(RemovePCStoresPass
                           revngc
                           RemovePCStoresPass.cpp)

target_link_libraries(RemovePCStoresPass revng::revngSupport)
