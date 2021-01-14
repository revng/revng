#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngSymbolFunctionBoundaries
  FunctionBoundariesFunctionSymbol.cpp)

target_link_libraries(revngSymbolFunctionBoundaries
  revngSupport
  revngBasicAnalyses)
