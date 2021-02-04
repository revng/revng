#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(revng-lift Main.cpp)


target_link_libraries(revng-lift
  dl
  m
  revngBasicAnalyses
  revngSupport
  revngFunctionCallIdentification
  revngLift
  ${LLVM_LIBRARIES})
