#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcBackend revngc CDecompilationPipe.cpp DecompileFunction.cpp
  DecompileToSingleFile.cpp DecompileToSingleFilePipe.cpp)

target_link_libraries(
  revngcBackend
  revngcInitModelTypes
  revngcRestructureCFG
  revngcSupport
  revngcTypeNames
  revng::revngABI
  revng::revngModel
  revng::revngPipes
  revng::revngPTML
  revng::revngSupport
  revng::revngYield
  ${LLVM_LIBRARIES})
