add_library(revngcExamplePass MODULE Pass.cpp)

target_link_libraries(revngcExamplePass ${REVNG_SUPPORT_LIBRARY})

install(TARGETS revngcExamplePass
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
