set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

add_library(revngcExamplePass SHARED Pass.cpp)

target_link_libraries(revngcExamplePass ${REVNG_SUPPORT_LIBRARY})

install(TARGETS revngcExamplePass
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
