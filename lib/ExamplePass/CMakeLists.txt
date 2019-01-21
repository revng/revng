set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

revng_add_analyses_library(revngcExamplePass SHARED Pass.cpp)

target_link_libraries(revngcExamplePass revng::revngSupport)
