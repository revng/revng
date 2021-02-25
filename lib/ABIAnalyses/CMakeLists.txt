#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngABIAnalyses
  ABIAnalysisPass.cpp
  DeadRegisterArgumentsOfFunction.cpp
  DeadReturnValuesOfFunctionCall.cpp
  UsedRegisterArgumentsOfFunction.cpp
  UsedReturnValuesOfFunctionCall.cpp)

target_link_libraries(revngABIAnalyses
  revngStackAnalysis
  revngSupport)
