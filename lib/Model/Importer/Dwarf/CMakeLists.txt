#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngModelImporterDwarf SHARED
  DwarfImporter.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES
  Object
  Support
  DebugInfoDWARF
  BinaryFormat)

target_link_libraries(revngModelImporterDwarf
  revngSupport
  revngModel
  revngModelPasses
  ${LLVM_LIBRARIES})
