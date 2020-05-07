#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(Decompiler revngc
  ASTBuildAnalysis.cpp
  DLACollapseIdentityAndInheritanceCC.cpp
  DLAComputeUpperMemberAccess.cpp
  DLACreateInterProceduralTypes.cpp
  DLACreateIntraProceduralTypes.cpp
  DLAHelpers.cpp
  DLAMakeInheritanceTree.cpp
  DLAPass.cpp
  DLAPruneLayoutNodesWithoutLayout.cpp
  DLARemoveTransitiveInheritanceEdges.cpp
  DLAStep.cpp
  DLATypeSystem.cpp
  CDecompiler.cpp
  CDecompilerAction.cpp
  CDecompilerBeautify.cpp
  CDecompilerPass.cpp
  DecompilationHelpers.cpp
  FuncDeclCreationAction.cpp
  GlobalDeclCreationAction.cpp
  IRASTTypeTranslation.cpp
  MarkForSerialization.cpp
  TypeDeclCreationAction.cpp)

target_link_libraries(Decompiler
  DecompilerResourceFinder
  FilterForDecompilation
  PHIASAPAssignmentInfo
  RestructureCFGPass
  RemoveNewPCCalls
  RemoveCpuLoopStore
  RemoveExceptionCalls
  RemoveLLVMAssumeCalls
  MakeEnvNull
  revngTypeShrinking
  revng::revngSupport
  clangTooling
  clangAST
  clangBasic
  clangFrontend
  clangSerialization
  ${LLVM_LIBRARIES})
