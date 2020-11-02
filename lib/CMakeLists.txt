#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

add_subdirectory(Decompiler)
add_subdirectory(DecompilerResourceFinder)
add_subdirectory(MakeEnvNull)
add_subdirectory(FilterForDecompilation)
add_subdirectory(PHIASAPAssignmentInfo)
add_subdirectory(RestructureCFGPass)
add_subdirectory(RemoveCpuLoopStore)
add_subdirectory(RemoveExceptionCalls)
add_subdirectory(RemoveLLVMAssumeCalls)
add_subdirectory(RemoveNewPCCalls)
add_subdirectory(TargetFunctionOption)
add_subdirectory(TypeShrinking)