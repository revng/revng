#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(
  revngYield SHARED Assembly/DisassemblyHelper.cpp
  Assembly/LLVMDisassemblerInterface.cpp)

set(DISASSEMBLER_ARCHITECTURE_SUPPORT "")
foreach(CURRENT_ARCHITECTURE AArch64 AMDGPU ARM Mips SystemZ X86)
  set(DISASSEMBLER_ARCHITECTURE_SUPPORT
      ${DISASSEMBLER_ARCHITECTURE_SUPPORT}
      "${CURRENT_ARCHITECTURE}Disassembler" "${CURRENT_ARCHITECTURE}Info"
      "${CURRENT_ARCHITECTURE}Desc")
endforeach(CURRENT_ARCHITECTURE)

llvm_map_components_to_libnames(
  LLVM_LIBRARIES
  Core
  MC
  MCDisassembler
  Support
  Target
  ${DISASSEMBLER_ARCHITECTURE_SUPPORT})

target_link_libraries(revngYield revngEarlyFunctionAnalysis revngModelPasses
                      revngSupport ${LLVM_LIBRARIES})
