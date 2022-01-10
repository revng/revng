#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcTypeNames revngc
  LLVMTypeNames.cpp
  ModelTypeNames.cpp)

target_link_libraries(revngcTypeNames
  revng::revngModel
  revng::revngSupport
  revngcSupport
  ${LLVM_LIBRARIES})