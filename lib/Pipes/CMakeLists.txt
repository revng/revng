#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngPipes
  SHARED
  CompileModulePipe.cpp
  FileContainer.cpp
  IsolatedKind.cpp
  Kinds.cpp
  LiftPipe.cpp
  LinkForTranslationPipe.cpp
  LinkSupportPipe.cpp
  LLVMContextWrapper.cpp
  ModelGlobal.cpp
  PipelineManager.cpp
  Pipes.cpp
  RootKind.cpp
)

llvm_map_components_to_libnames(LLVM_LIBRARIES
  Core
  IPO
  MC
  Support
  X86AsmParser
  X86CodeGen
  X86Desc
  X86Info
)

target_link_libraries(revngPipes
  revngEarlyFunctionAnalysis
  revngLift
  revngPipeline
  revngSupport
  ${LLVM_LIBRARIES}
)
