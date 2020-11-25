#
# Copyright rev.ng Srls. See LICENSE.md for details.
#

revng_add_analyses_library(FilterForDecompilation revngc
  FilterForDecompilationPass.cpp)

target_link_libraries(FilterForDecompilation
  revng::revngSupport
  ${LLVM_LIBRARIES})

