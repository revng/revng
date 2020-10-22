revng_add_analyses_library(RemoveSwitchPass
                           revngc
                           RemoveSwitchPass.cpp
                           )

target_link_libraries(RemoveSwitchPass revng::revngSupport)
