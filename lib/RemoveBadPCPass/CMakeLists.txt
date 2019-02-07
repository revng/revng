set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

revng_add_analyses_library(RemoveBadPCPass SHARED
                           RemoveBadPCPass.cpp
                           )

target_link_libraries(RemoveBadPCPass revng::revngSupport)
