#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngABIAnalyses
  ABIAnalysisPass.cpp
  DeadReturnValuesOfFunctionCall.cpp
  DeadRegisterArgumentsOfFunction.cpp)

target_link_libraries(revngABIAnalyses
  revngStackAnalysis
  revngSupport)
