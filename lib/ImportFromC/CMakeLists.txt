#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

# ImportFromC

revng_add_analyses_library(revngcImportFromCAnalysis revngc HeaderToModel.cpp
                           ImportFromCAnalysis.cpp)

target_link_libraries(
  revngcImportFromCAnalysis
  revngcTypeNames
  revng::revngModel
  revng::revngSupport
  revng::revngPipeline
  revng::revngPipes
  revng::revngPTML
  revng::revngBasicAnalyses
  revngcModelToHeader
  clangBasic
  clangAST
  clangDriver
  clangSerialization
  clangFrontend
  clangTooling
  ${LLVM_LIBRARIES})
