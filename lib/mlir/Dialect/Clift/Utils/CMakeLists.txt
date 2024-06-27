#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library(MLIRCliftUtils SHARED revngc ImportModel.cpp)

target_link_libraries(
  MLIRCliftUtils PUBLIC MLIRCliftDialect MLIRLLVMDialect revng::revngModel
                        revng::revngEarlyFunctionAnalysis)
