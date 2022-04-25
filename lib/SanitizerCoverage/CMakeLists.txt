#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngSanitizerCoverage
  SanitizerCoverage.cpp)

target_link_libraries(revngSanitizerCoverage revngModel revngSupport ${LLVM_LIBRARIES})
