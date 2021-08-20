#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(ThreadSafeClangTooling revngc
  ThreadSafeClangTooling.cpp)

target_link_libraries(ThreadSafeClangTooling
  clangAST
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  ${LLVM_LIBRARIES})
