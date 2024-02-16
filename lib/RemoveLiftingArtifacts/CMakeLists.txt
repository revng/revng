#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcRemoveLiftingArtifacts revngc PromoteInitCSVToUndef.cpp
  RemoveLiftingArtifacts.cpp MakeSegmentRefPass.cpp MakeSegmentRefPipe.cpp)

target_link_libraries(
  revngcRemoveLiftingArtifacts
  revngcSupport
  revng::revngModel
  revng::revngSupport
  revng::revngPipes
  revng::revngLift
  ${LLVM_LIBRARIES})
