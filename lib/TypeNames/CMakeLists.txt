#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngTypeNames DependencyGraph.cpp LLVMTypeNames.cpp ModelTypeNames.cpp
  TypePrinters.cpp)

target_link_libraries(revngTypeNames revngSupport revngABI revngModel revngPTML
                      ${LLVM_LIBRARIES})
