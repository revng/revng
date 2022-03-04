#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcRemoveExtractValues revngc RemoveExtractValuesPass.cpp
  RestoreExtractValuesPass.cpp)

target_link_libraries(revngcRemoveExtractValues revngcSupport
                      revng::revngSupport ${LLVM_LIBRARIES})
