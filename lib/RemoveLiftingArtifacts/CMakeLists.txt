#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngRemoveLiftingArtifacts PromoteInitCSVToUndef.cpp
  RemoveLiftingArtifacts.cpp)

target_link_libraries(revngRemoveLiftingArtifacts revngSupport revngModel
                      revngLift ${LLVM_LIBRARIES})
