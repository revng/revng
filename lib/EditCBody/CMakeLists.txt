#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# EditCBody

revng_add_analyses_library_internal(revngEditCBodyAnalysis ClangParse.cpp
                                    CliftFlatten.cpp EditCBodyAnalysis.cpp)

target_link_libraries(
  revngEditCBodyAnalysis
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
