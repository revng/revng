#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# AnnotateCBody

revng_add_analyses_library_internal(
  revngAnnotateCBodyAnalysis ClangParse.cpp CliftFlatten.cpp
  AnnotateCBodyAnalysis.cpp EditByNameAnalysis.cpp ModelEdits.cpp)

target_link_libraries(
  revngAnnotateCBodyAnalysis
  revngClangToModel
  revngClift
  revngCliftEmitC
  revngCliftImportModel
  revngModel
  revngPTML
  revngSupport
  clangAST
  clangBasic
  clangDriver
  clangFrontend
  clangSerialization
  clangTooling
  ${LLVM_LIBRARIES})
