#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  revngPipes
  SHARED
  IRHelpers.cpp
  PipelineManager.cpp
  Pipes.cpp
  RootKind.cpp
  TaggedFunctionKind.cpp
  ApplyDiffAnalysis.cpp)

llvm_map_components_to_libnames(
  LLVM_LIBRARIES
  AggressiveInstCombine
  Analysis
  AsmParser
  BitWriter
  CodeGen
  Core
  Coroutines
  Extensions
  GlobalISel
  IPO
  InstCombine
  Instrumentation
  MC
  Passes
  Remarks
  ScalarOpts
  Support
  Target
  TransformUtils
  Vectorize
  X86AsmParser
  X86CodeGen
  X86Desc
  X86Info)

target_link_libraries(revngPipes revngPipeline revngModel revngSupport
                      ${LLVM_LIBRARIES})
