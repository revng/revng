#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(Decompiler revngc
  AddSCEVBarrierPass.cpp
  ASTBuildAnalysis.cpp
  CDecompiler.cpp
  CDecompilerAction.cpp
  CDecompilerBeautify.cpp
  CDecompilerPass.cpp
  DecompilationHelpers.cpp
  FuncDeclCreationAction.cpp
  GlobalDeclCreationAction.cpp
  IRASTTypeTranslation.cpp
  MarkForSerialization.cpp
  SCEVBaseAddressExplorer.cpp
  TypeDeclCreationAction.cpp)

target_link_libraries(Decompiler
  FilterForDecompilation
  LivenessAnalysis
  PHIASAPAssignmentInfo
  RestructureCFGPass
  RemoveNewPCCalls
  RemoveCpuLoopStore
  RemoveExceptionCalls
  RemoveLLVMAssumeCalls
  RemoveLLVMDbgIntrinsics
  MakeEnvNull
  ThreadSafeClangTooling
  Utils
  revng::revngModel
  revng::revngSupport
  revng::revngTypeShrinking
  clangTooling
  clangAST
  clangBasic
  clangFrontend
  clangSerialization
  DataLayoutAnalysis
  ${LLVM_LIBRARIES})
