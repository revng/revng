#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_analyses_library_internal(revngModelImporterBinary
  BinaryImporter.cpp
  ELFImporter.cpp
  MachOImporter.cpp
  PECOFFImporter.cpp
)

llvm_map_components_to_libnames(LLVM_LIBRARIES Object)
target_link_libraries(revngModelImporterBinary
  revngModel
  revngModelImporterDwarf
  revngABI
  ${LLVM_LIBRARIES}
)
