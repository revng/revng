#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

macro(add_tool tool_name sources)
  revng_add_executable(${tool_name} ${sources} Common.cpp)

  llvm_map_components_to_libnames(LLVM_LIBRARIES Core Support)
  target_link_libraries(
    ${tool_name}
    revngEarlyFunctionAnalysis
    revngLift
    revngModel
    revngPipes
    revngSupport
    revngYield
    ${LLVM_LIBRARIES})
endmacro()

add_tool(revng-yield-assembly-html AssemblyHTML.cpp)
add_tool(revng-yield-assembly-plain AssemblyPlain.cpp)
