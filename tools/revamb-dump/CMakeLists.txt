#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

add_executable(revamb-dump
  CollectCFG.cpp
  CollectFunctionBoundaries.cpp
  CollectNoreturn.cpp
  IsolateFunctions.cpp
  Main.cpp)

target_link_libraries(revamb-dump
  revngSupport
  revngDebugHelper
  revngStackAnalysis
  revngBasicAnalyses
  ${LLVM_LIBRARIES})

add_custom_command(TARGET revamb-dump POST_BUILD VERBATIM
  COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:revamb-dump>" "${CMAKE_BINARY_DIR}/")

install(TARGETS revamb-dump
  RUNTIME DESTINATION bin)
