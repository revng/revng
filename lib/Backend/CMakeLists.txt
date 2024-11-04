#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcBackend
  revngc
  ALAPVariableDeclaration.cpp
  DecompilePipe.cpp
  DecompileFunction.cpp
  DecompileToDirectoryPipe.cpp
  DecompileToSingleFile.cpp
  DecompileToSingleFilePipe.cpp)

target_link_libraries(
  revngcBackend
  revngcHelpersToHeader
  revngcInitModelTypes
  revngcModelToHeader
  revngcRestructureCFG
  revngcSupport
  revngcTypeNames
  revng::revngABI
  revng::revngModel
  revng::revngPipes
  revng::revngPTML
  revng::revngYield
  ${LLVM_LIBRARIES})
