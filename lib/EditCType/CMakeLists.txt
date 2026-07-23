#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# EditCType

revng_add_analyses_library_internal(revngEditCTypeAnalysis EditCType.cpp
                                    EditCTypeAnalysis.cpp)

target_link_libraries(
  revngEditCTypeAnalysis
  revngABI
  revngBasicAnalyses
  revngClangToModel
  revngCliftEmitC
  revngCliftImportModel
  revngModel
  revngPTML
  revngPipeline
  revngPipes
  revngSupport
  clangAST
  clangBasic
  clangDriver
  clangFrontend
  clangSerialization
  clangTooling
  ${LLVM_LIBRARIES})
