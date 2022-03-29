#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcRemoveLiftingArtifacts
  revngc
  FilterForDecompilationPass.cpp
  MakeEnvNull.cpp
  RemoveCpuLoopStorePass.cpp
  RemoveExceptionCallsPass.cpp
  RemoveLLVMDbgIntrinsicsPass.cpp
  RemoveNewPCCallsPass.cpp)

target_link_libraries(revngcRemoveLiftingArtifacts revngcSupport
                      revng::revngModel revng::revngSupport ${LLVM_LIBRARIES})
