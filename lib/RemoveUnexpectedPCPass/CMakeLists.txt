set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

revng_add_analyses_library(RemoveUnexpectedPCPass SHARED
                           RemoveUnexpectedPCPass.cpp)

target_link_libraries(RemoveUnexpectedPCPass revng::revngSupport)
