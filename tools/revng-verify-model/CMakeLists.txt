#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_executable(revng-verify-model VerifyModelMain.cpp)

target_link_libraries(revng-verify-model
  revngModel
  revngSupport
  ${LLVM_LIBRARIES}
  )
