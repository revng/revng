add_library(BasicAnalyses
  STATIC
  FunctionCallIdentification.cpp
  GeneratedCodeBasicInfo.cpp
  ReachingDefinitionsPass.cpp)
target_link_libraries(BasicAnalyses Support)
