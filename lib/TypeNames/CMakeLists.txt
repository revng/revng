#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library(
  revngcTypeNames revngc DependencyGraph.cpp LLVMTypeNames.cpp
  ModelTypeNames.cpp TypePrinters.cpp)

target_link_libraries(revngcTypeNames revngcSupport revng::revngABI
                      revng::revngModel revng::revngPTML ${LLVM_LIBRARIES})
