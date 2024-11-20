#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngRemoveLiftingArtifacts CleanupIRPass.cpp PromoteInitCSVToUndef.cpp
  RemoveLiftingArtifacts.cpp MakeSegmentRefPass.cpp MakeSegmentRefPipe.cpp)

target_link_libraries(revngRemoveLiftingArtifacts revngSupport revngModel
                      revngPipes revngLift ${LLVM_LIBRARIES})
