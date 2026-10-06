#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngTypeShrinking EarlyTypeShrinking.cpp RewritePlans.cpp TypeShrinking.cpp
  TypeShrinkingAnalysis.cpp)

target_link_libraries(revngTypeShrinking revngSupport ${LLVM_LIBRARIES})
