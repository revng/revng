#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(PHIASAPAssignmentInfo revngc
  PHIASAPAssignmentInfo.cpp)

target_link_libraries(PHIASAPAssignmentInfo
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})
