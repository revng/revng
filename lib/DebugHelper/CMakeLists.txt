add_library(revngDebugHelper STATIC
  DebugHelper.cpp)

install(TARGETS revngDebugHelper
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
