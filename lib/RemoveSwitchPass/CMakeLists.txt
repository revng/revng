set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

revng_add_analyses_library(RemoveSwitchPass SHARED
                           RemoveSwitchPass.cpp
                           )

target_link_libraries(RemoveSwitchPass revng::revngSupport)
