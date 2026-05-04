#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngHelpersToHeader HelpersToHeader.cpp
                                    HelpersToHeaderPipe.cpp)

target_link_libraries(
  revngHelpersToHeader
  revngTypeNames
  revngSupport
  revngModel
  revngPipeline
  revngPipes
  revngPTML
  ${LLVM_LIBRARIES})
