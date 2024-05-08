#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#

revng_add_library_internal(revngModelImporterDebugInfo SHARED DwarfImporter.cpp
                           PDBImporter.cpp)

llvm_map_components_to_libnames(
  LLVM_LIBRARIES
  Object
  Support
  DebugInfoDWARF
  DebugInfoCodeView
  DebugInfoMSF
  DebugInfoPDB
  BinaryFormat)

target_link_libraries(revngModelImporterDebugInfo revngSupport revngModel
                      revngModelPasses ${LLVM_LIBRARIES})
