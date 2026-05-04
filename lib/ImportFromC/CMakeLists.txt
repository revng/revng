#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# ImportFromC

revng_add_analyses_library_internal(revngImportFromCAnalysis HeaderToModel.cpp
                                    ImportFromCAnalysis.cpp)

target_link_libraries(
  revngImportFromCAnalysis
  revngABI
  revngBasicAnalyses
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
