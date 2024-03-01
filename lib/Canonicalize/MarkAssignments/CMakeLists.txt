#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcMarkAssignments revngc AddAssignmentMarkerPass.cpp LivenessAnalysis.cpp
  MarkAssignments.cpp)

target_link_libraries(revngcMarkAssignments revngcInitModelTypes revngcSupport
                      revng::revngModel revng::revngSupport ${LLVM_LIBRARIES})
