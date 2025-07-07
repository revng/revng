#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(
  MLIRCliftUtils
  SHARED
  CBackend.cpp
  CSemantics.cpp
  CTarget.cpp
  ImportLLVM.cpp
  ImportModel.cpp
  ModelVerify.cpp)

target_link_libraries(
  MLIRCliftUtils
  PUBLIC MLIRCliftDialect
         MLIRLLVMDialect
         revngModel
         revngEarlyFunctionAnalysis
         revngPTML
         revngTypeNames
         revngRestructureCFG
         revngLocalVariables)
