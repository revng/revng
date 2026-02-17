#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(model-import-sqlite Main.cpp)

llvm_map_components_to_libnames(LLVM_LIBRARIES Support)
target_link_libraries(model-import-sqlite revngModel revngSupport
                      ${LLVM_LIBRARIES} sqlite3)
