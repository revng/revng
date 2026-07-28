#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

# ClangToModel

revng_add_library_internal(revngClangToModel SHARED QualTypeToModel.cpp
                           CompileFlags.cpp)

target_link_libraries(
  revngClangToModel
  revngModel
  revngSupport
  clangAST
  clangBasic
  clangDriver
  ${LLVM_LIBRARIES})
