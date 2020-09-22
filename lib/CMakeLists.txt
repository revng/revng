#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

add_subdirectory(AdjustStackPointer)
add_subdirectory(Decompiler)
add_subdirectory(Liveness)
add_subdirectory(MakeEnvNull)
add_subdirectory(FilterForDecompilation)
add_subdirectory(PHIASAPAssignmentInfo)
add_subdirectory(PromoteStackPointer)
add_subdirectory(RestructureCFGPass)
add_subdirectory(RemoveCpuLoopStore)
add_subdirectory(RemoveExceptionCalls)
add_subdirectory(RemoveLLVMAssumeCalls)
add_subdirectory(RemoveLLVMDbgIntrinsics)
add_subdirectory(RemoveNewPCCalls)
add_subdirectory(TargetFunctionOption)
add_subdirectory(ThreadSafeClangTooling)
add_subdirectory(ValueManipulationAnalysis)
