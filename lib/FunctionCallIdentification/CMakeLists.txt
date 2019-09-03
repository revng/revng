#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngFunctionCallIdentification
  STATIC
  FunctionCallIdentification.cpp)

target_link_libraries(revngFunctionCallIdentification
  revngSupport
  revngBasicAnalyses)
