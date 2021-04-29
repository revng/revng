#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(RemoveLLVMDbgIntrinsics
  revngc
	RemoveLLVMDbgIntrinsicsPass.cpp)

target_link_libraries(RemoveLLVMDbgIntrinsics
  revng::revngModel
  revng::revngSupport)

