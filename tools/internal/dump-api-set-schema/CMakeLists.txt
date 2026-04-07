#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

revng_add_executable(internal-dump-api-set-schema Main.cpp)

target_link_libraries(internal-dump-api-set-schema revngSupport
                      ${LLVM_LIBRARIES})
