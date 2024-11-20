#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngBackend
  ALAPVariableDeclaration.cpp
  DecompilePipe.cpp
  DecompileFunction.cpp
  DecompileToDirectoryPipe.cpp
  DecompileToSingleFile.cpp
  DecompileToSingleFilePipe.cpp)

target_link_libraries(
  revngBackend
  revngHelpersToHeader
  revngInitModelTypes
  revngModelToHeader
  revngRestructureCFG
  revngSupport
  revngTypeNames
  revngABI
  revngModel
  revngPipes
  revngPTML
  revngYield
  ${LLVM_LIBRARIES})
