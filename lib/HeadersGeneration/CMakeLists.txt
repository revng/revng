#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(revngcModelToHeader revngc ModelToHeader.cpp)

target_link_libraries(revngcModelToHeader
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})

revng_add_analyses_library(revngcModelToHeaderPass revngc ModelToHeaderPass.cpp)

target_link_libraries(revngcModelToHeaderPass
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES}
  revngcModelToHeader)
