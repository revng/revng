#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngABIAnalyses
  ABIAnalysis.cpp
  ABIAnalysisPass.cpp
  DeadRegisterArgumentsOfFunction.cpp
  DeadReturnValuesOfFunctionCall.cpp
  RegisterArgumentsOfFunctionCall.cpp
  UsedArgumentsOfFunction.cpp
  UsedReturnValuesOfFunction.cpp
  UsedReturnValuesOfFunctionCall.cpp)

target_link_libraries(revngABIAnalyses
  revngSupport)
