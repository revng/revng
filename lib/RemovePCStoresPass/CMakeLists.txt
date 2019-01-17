set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

add_library(RemovePCStoresPass SHARED
            RemovePCStoresPass.cpp
            )

target_link_libraries(RemovePCStoresPass ${REVNG_SUPPORT_LIBRARY})

install(TARGETS RemovePCStoresPass
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
