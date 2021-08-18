#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngDwarfImporter SHARED
  DwarfImporter.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES DebugInfoDWARF BinaryFormat)

target_link_libraries(revngDwarfImporter
  ${LLVM_LIBRARIES})