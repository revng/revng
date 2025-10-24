#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngHelperArgumentsAnalysis
  AnalyzeHelperArguments.cpp
  Annotation.cpp
  ArgumentUsageAnalysis.cpp
  Context.cpp
  CPULoopExitPass.cpp
  CPUStateUsage.cpp
  FixHelpers.cpp
  SlimDownHelpersModule.cpp
  Value.cpp)

target_link_libraries(revngHelperArgumentsAnalysis revngLift revngSupport
                      ${LLVM_LIBRARIES})
