#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngBasicAnalyses
  EmptyNewPC.cpp
  RemoveDbgMetadata.cpp
  GeneratedCodeBasicInfo.cpp
  MaterializedValue.cpp
  RemoveNewPCCalls.cpp
  RemoveHelperCalls.cpp
  ShrinkInstructionOperandsPass.cpp)

target_link_libraries(revngBasicAnalyses revngModel revngSupport
                      ${LLVM_LIBRARIES})
