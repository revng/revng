#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngPipes SHARED
  FileContainer.cpp
  CompileModulePipe.cpp
  LiftPipe.cpp
  LinkSupportPipe.cpp
  LinkForTranslationPipe.cpp
  RevngPipes.cpp
  RootKind.cpp
  IsolatedKind.cpp
  PipelineManager.cpp
  Kinds.cpp
  LLVMContextWrapper.cpp
  ModelGlobal.cpp
  )

llvm_map_components_to_libnames(LLVM_LIBRARIES MC Support Core x86codegen x86desc x86info x86asmparser ipo)

target_link_libraries(revngPipes
  revngSupport
  revngLift
  revngPipeline
  revngEarlyFunctionAnalysis
  ${LLVM_LIBRARIES})
