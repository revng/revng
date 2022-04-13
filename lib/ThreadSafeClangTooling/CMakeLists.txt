#
# Copyright rev.ng Labs Srl. See LICENSE.md for details.
#

revng_add_analyses_library(revngcThreadSafeClangTooling revngc
                           ThreadSafeClangTooling.cpp)

target_link_libraries(
  revngcThreadSafeClangTooling
  clangAST
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  ${LLVM_LIBRARIES})
