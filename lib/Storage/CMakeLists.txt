#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

find_package(AWSSDK REQUIRED COMPONENTS s3)

revng_add_library_internal(revngStorage SHARED StorageClient.cpp
                           S3StorageClient.cpp LocalStorageClient.cpp Path.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Core Support)

target_link_libraries(revngStorage revngSupport ${AWSSDK_LINK_LIBRARIES}
                      ${LLVM_LIBRARIES})
