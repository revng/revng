#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcPipes revngc CDecompilationPipe.cpp)

target_link_libraries(
  revngcPipes
  revngcBackend
  revng::revngModel
  revng::revngSupport
  revng::revngPipeline
  revng::revngPipes
  ${LLVM_LIBRARIES})
