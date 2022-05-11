#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcRemoveLiftingArtifacts revngc
                           PromoteInitCSVToUndef.cpp RemoveLiftingArtifacts.cpp)

target_link_libraries(
  revngcRemoveLiftingArtifacts revngcSupport revng::revngModel
  revng::revngSupport revng::revngPipes ${LLVM_LIBRARIES})
