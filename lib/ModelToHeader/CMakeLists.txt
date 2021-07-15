#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(ModelToHeader revngc ModelToHeader.cpp)

target_link_libraries(ModelToHeader
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES})

revng_add_analyses_library(ModelToHeaderPass revngc ModelToHeaderPass.cpp)

target_link_libraries(ModelToHeaderPass
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES}
  ModelToHeader)

revng_add_executable(revng-model-to-header ModelToHeaderMain.cpp)

target_link_libraries(revng-model-to-header
  revng::revngModel
  revng::revngSupport
  ${LLVM_LIBRARIES}
  ModelToHeader)
