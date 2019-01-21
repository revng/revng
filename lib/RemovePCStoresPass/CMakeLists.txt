set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

revng_add_analyses_library(RemovePCStoresPass SHARED
                           RemovePCStoresPass.cpp
                           )

target_link_libraries(RemovePCStoresPass revng::revngSupport)
