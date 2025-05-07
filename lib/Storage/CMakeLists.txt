#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_library_internal(revngStorage SHARED StorageClient.cpp Path.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Core Support)

target_link_libraries(revngStorage revngSupport ${LLVM_LIBRARIES})
