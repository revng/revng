#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

# ImportModelFromC

revng_add_analyses_library(revngcImportModelFromCAnalysis revngc
                           HeaderToModel.cpp ImportModelFromCAnalysis.cpp)

target_link_libraries(
  revngcImportModelFromCAnalysis
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
